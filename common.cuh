#pragma once

constexpr int LOG_NUM_BANKS = 5; // NUM_BANKS = 32; // unused
constexpr int warpSize = 32;
constexpr unsigned MASK_ALL = 0xffff'ffffu;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    std::cout << "GPU assert failed" << std::endl;
    if (abort) exit(code);
  }
}


static __device__ float warp_scan(float value) {
  const auto lane = threadIdx.x & (warpSize - 1);
  for (auto s = 1; s < warpSize; s *= 2) {
    auto const tmp = __shfl_up_sync(MASK_ALL, value, s);
    if (lane >= s)
      value += tmp;
  }
  return value;
}

template <int blockSize, int itemsPerThread>
__device__ void block_scan(float (&thread_data)[itemsPerThread], float *warp_sums) {
  float total_warp_sum{0};
  auto warp = threadIdx.x / warpSize;
  auto lane = threadIdx.x & (warpSize - 1);
  for (int i = 0; i < itemsPerThread; i++) {
    auto pref_within_warp = warp_scan(thread_data[i]);
    auto warp_sum = __shfl_sync(MASK_ALL, pref_within_warp, warpSize-1);
    thread_data[i] = total_warp_sum + pref_within_warp;
    total_warp_sum += warp_sum;
  }
  // write warp sums
  if (lane == warpSize - 1)
     warp_sums[warp] = total_warp_sum;
  __syncthreads();

  // scan warp sums in a separate warp
  if (warp == 0)
    warp_sums[lane] = warp_scan(warp_sums[lane]);
  __syncthreads();

  auto prefix = (warp > 0) ? warp_sums[warp-1] : 0.f;
  for (int i = 0; i < itemsPerThread; i++) {
    thread_data[i] += prefix;
  }
}

template <int blockSize, int itemsPerThread>
__device__ void block_load(int chunk, float const *in, size_t input_size, float * shared)
{
  // load into shared memory
  constexpr int chunkSize = blockSize*itemsPerThread;
  int const tid = threadIdx.x;

  for (int i = 0; i < itemsPerThread; i++) {
    int idx = chunk*chunkSize + i*blockSize + tid;
    shared[i*blockSize + tid] = (idx < input_size) ? in[idx] : 0.f;
  }
}

template <int blockSize, int itemsPerThread>
__device__ void block_load(int chunk, float const *in, size_t input_size,
                           float (&thread_data)[itemsPerThread],
                           float * shared) {
  // // load into shared memory
  // constexpr int chunkSize = blockSize*itemsPerThread;
  // int const tid = threadIdx.x;

  // for (int i = 0; i < itemsPerThread; i++) {
  //   int idx = chunk*chunkSize + i*blockSize + tid;
  //   shared[i*blockSize + tid] = (idx < input_size) ? in[idx] : 0.f;
  // }
  block_load<blockSize,itemsPerThread>(chunk, in, input_size, shared);
  __syncthreads();

  // load from shared memory into thread local data
  int const tid = threadIdx.x;
  auto const warp = tid / warpSize;
  auto lane = tid & (warpSize-1);
  for (int i = 0; i < itemsPerThread; i++) {
    auto idx = warp*warpSize*itemsPerThread + lane + i*warpSize;
    thread_data[i] = shared[idx];
  }
}

template <int blockSize, int itemsPerThread>
__device__ void block_store(int chunk, float *out, size_t input_size, float * shared)
{
  constexpr int chunkSize = blockSize*itemsPerThread;
  int const tid = threadIdx.x;
  for (int i = 0; i < itemsPerThread; i++) {
    int pos_glob = chunk*chunkSize + i*blockSize + tid;
    int pos = i*blockSize + tid;
    if (pos_glob < input_size) {
      out[pos_glob] = shared[pos];
    }
  }
}

template <int blockSize, int itemsPerThread>
__device__ void block_store(int chunk, float *out, size_t input_size,
                            float (&thread_data)[itemsPerThread],
                            float * shared) {
  // constexpr int chunkSize = blockSize*itemsPerThread;
  int const tid = threadIdx.x;
  auto const warp = tid / warpSize;
  auto lane = threadIdx.x & (warpSize-1);
  for (int i = 0; i < itemsPerThread; i++) {
    auto idx = warp*warpSize*itemsPerThread + lane + i*warpSize;
    shared[idx] = thread_data[i];
  }
  __syncthreads();

  block_store<blockSize, itemsPerThread>(chunk, out, input_size, shared);
  // for (int i = 0; i < itemsPerThread; i++) {
  //   int pos_glob = chunk*chunkSize + i*blockSize + tid;
  //   int pos = i*blockSize + tid;
  //   if (pos_glob < input_size) {
  //     out[pos_glob] = shared[pos];
  //   }
  // }
}
