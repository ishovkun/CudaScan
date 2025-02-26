#pragma once

constexpr int LOG_NUM_BANKS = 5; // NUM_BANKS = 32; // unused
constexpr int WARP_SIZE = 32;
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
  const auto lane = threadIdx.x & (WARP_SIZE - 1);
  for (auto s = 1; s < WARP_SIZE; s *= 2) {
    auto const tmp = __shfl_up_sync(MASK_ALL, value, s);
    if (lane >= s)
      value += tmp;
  }
  return value;
}

template <int blockSize, int itemsPerThread>
static __device__ void block_scan(float (&thread_data)[itemsPerThread], float *warp_sums) {
  float total_warp_sum{0};
  auto warp = threadIdx.x / WARP_SIZE;
  auto lane = threadIdx.x & (WARP_SIZE - 1);
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
