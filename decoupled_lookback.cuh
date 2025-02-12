#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>
#include "common.cuh"
#include <cuda_runtime.h>
#include <cuda/atomic>

enum class TileStatus : int32_t {
  unavailable,
  local_prefix_available,
  global_prefix_available,
};

struct ScanState {
  TileStatus status{TileStatus::unavailable}; // 4 bytes
  float sum{0.f}; // 4 bytes
};

template <int blockSize>
__device__ float block_scan(int thread_value)
{
  constexpr int shmem_size = 1 + blockSize/WARP_SIZE;
  __shared__ float warp_sums[shmem_size];

  auto tid = threadIdx.x;
  if (tid < shmem_size)
    warp_sums[tid] = 0.f;

  auto thread_prefix_within_warp = warp_scan(thread_value);
  __syncthreads();

  auto warp = threadIdx.x / WARP_SIZE;
  auto lane = threadIdx.x & (WARP_SIZE-1);
  if (lane == WARP_SIZE-1)
    warp_sums[1+warp] = thread_prefix_within_warp;
  __syncthreads();
  if (warp == 0)

    warp_sums[1+lane] = warp_scan(warp_sums[1+lane]);
  __syncthreads();

  return thread_prefix_within_warp + warp_sums[warp];
}

// static __device__ void store_status(volatile ScanState *ptr, ScanState state)
// {
//   *reinterpret_cast<volatile uint64_t*>(ptr) = *reinterpret_cast<const uint64_t*>(&state);
// }

static __device__ void store_status(ScanState *ptr, ScanState state)
{
  auto * src = reinterpret_cast<unsigned long long int*>(&state);
  auto * dst = reinterpret_cast<unsigned long long int*>(ptr);
  atomicExch(dst, *src);
  // auto* dst = reinterpret_cast<cuda::atomic<uint64_t, cuda::thread_scope_device>*>(ptr);
  // auto const* src = reinterpret_cast<const uint64_t*>(&state);
  // dst->store(*src, cuda::memory_order_release);
}


static __device__ ScanState load_status(ScanState const *ptr)
{
  ScanState state;
  auto const *src = reinterpret_cast<const cuda::atomic<uint64_t, cuda::thread_scope_device>*>(ptr);
  auto dst = reinterpret_cast<uint64_t*>(&state);
  *dst = src->load(cuda::memory_order_relaxed);
  return state;
}
// static __device__ ScanState load_status(volatile ScanState const *ptr)
// {
//   ScanState state;
//   *reinterpret_cast<uint64_t*>(&state) = *reinterpret_cast<volatile uint64_t const*>(ptr);
//   return state;
// }

__device__ float lookback(uint32_t chunk, ScanState* states)
{
  float prefix_sum{0.f};
  for(int prev_chunk = chunk - 1; prev_chunk >= 0; prev_chunk--) {
    ScanState prev_state{TileStatus::unavailable, 0.f};
    while (prev_state.status == TileStatus::unavailable) {
      prev_state = load_status(&states[prev_chunk]);
    }
    prefix_sum += prev_state.sum;
    if (prev_state.status == TileStatus::global_prefix_available)
      break;
  }
  return prefix_sum;
}

template <int blockSize>
__global__ void scan_decoupled_lookback(float* out, float const* in, int n,
                                        ScanState* states,
                                        uint* tile_counter)
{
  __shared__ uint sh_chunk;

  auto tid = threadIdx.x;
  if (tid == 0)
    sh_chunk = atomicAdd(tile_counter, 1u);
  __syncthreads();

  auto const chunk = sh_chunk;
  auto const i = chunk*blockSize + tid;
  auto const thread_value = (i < n) ? in[i] : 0.f;
  auto const local_sum = block_scan<blockSize>(thread_value);
  bool const last_thread_in_block = (tid == blockSize - 1);

  if (last_thread_in_block) {
    store_status(&states[chunk], ScanState{TileStatus::local_prefix_available, local_sum});
  }

  auto prefix_sum = lookback(chunk, states);
  auto aggregate = local_sum + prefix_sum;

  if (last_thread_in_block) {
    store_status(&states[chunk], ScanState{TileStatus::global_prefix_available, aggregate});
  }
  if (i < n) {
    out[i] = aggregate;
  }
}


template<int blockSize>
void scan_single_pass(thrust::device_vector<float> & in,
                      thrust::device_vector<float> & out)
{
  uint32_t* tile_counter;
  cudaMalloc(&tile_counter, sizeof(uint32_t));
  cudaMemset(tile_counter, 0, sizeof(unsigned int));

  // constexpr int blockSize = 32;
  int64_t numBlocks = (in.size() + blockSize - 1) / blockSize;

  thrust::device_vector<ScanState> states(numBlocks);
  thrust::fill(states.begin(), states.end(), ScanState{TileStatus::unavailable, 0.f});
  // std::cout << "size of TileStatus = " << sizeof(TileStatus) << std::endl;
  // std::cout << "size of ScanState = " << sizeof(ScanState) << std::endl;
  // std::cout << "num chunks = " << numBlocks << std::endl;
  // exit(0);


  scan_decoupled_lookback<blockSize><<<numBlocks, blockSize>>>
      (out.data().get(), in.data().get(), in.size(), states.data().get(), tile_counter);

  cudaFree(tile_counter);
}
