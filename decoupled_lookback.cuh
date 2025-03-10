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

__host__ __device__ inline int to_int(TileStatus status)
{
  return static_cast<int>(status);
}

struct ScanState {
  TileStatus status{TileStatus::unavailable}; // 4 bytes
  float sum{0.f}; // 4 bytes
  // ScanState& operator=(const ScanState& other) = default;
};

template <int blockSize>
__device__ float block_scan(int thread_value)
{
  constexpr int shmem_size = 1 + blockSize/warpSize;
  __shared__ float warp_sums[shmem_size];

  auto tid = threadIdx.x;
  if (tid < shmem_size)
    warp_sums[tid] = 0.f;

  auto thread_prefix_within_warp = warp_scan(thread_value);
  __syncthreads();

  auto warp = threadIdx.x / warpSize;
  auto lane = threadIdx.x & (warpSize-1);
  if (lane == warpSize-1)
    warp_sums[1+warp] = thread_prefix_within_warp;
  __syncthreads();
  if (warp == 0)
    warp_sums[1+lane] = warp_scan(warp_sums[1+lane]);
  __syncthreads();

  return thread_prefix_within_warp + warp_sums[warp];
}

static __device__ void store_status(ScanState *ptr, ScanState state)
{
  // auto *dst = reinterpret_cast<cuda::atomic<uint64_t, cuda::thread_scope_device>*>(ptr);
  // auto *src = reinterpret_cast<uint64_t*>(&state);
  // dst->store(*src, cuda::memory_order_relaxed);

  // this seems to be the fastest
  auto * src = reinterpret_cast<const unsigned long long int*>(&state);
  auto * dst = reinterpret_cast<unsigned long long int*>(ptr);
  atomicExch(dst, *src);

  // auto * src = reinterpret_cast<uint64_t*>(&state);
  // auto * dst = reinterpret_cast<uint64_t*>(ptr);
  // __stcg(dst, *src); // cache only in L2
  // __stwt(dst, *src); // write through
}

// template<typename thread_scope = cuda::thread_scope_device>
static __device__ ScanState load_status(ScanState const *ptr)
{
  ScanState state;
  auto const *src = reinterpret_cast<const cuda::atomic<uint64_t, cuda::thread_scope_device>*>(ptr);
  auto dst = reinterpret_cast<uint64_t*>(&state);
  *dst = src->load(cuda::memory_order_relaxed);

  // auto const *src = reinterpret_cast<uint64_t const*>(ptr);
  // auto dst = reinterpret_cast<uint64_t*>(&state);
  // *dst = __ldcv(src); // do not cache
  // *dst = __ldcg(src); // cache in L2, not L1

  return state;
}

__device__ float serial_lookback(uint32_t chunk, ScanState* states)
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

static __device__ __forceinline__ ScanState shfl_up_sync_status(ScanState const& state, int delta)
{
  auto *value = reinterpret_cast<unsigned long long int const*>(&state);
  auto result = __shfl_up_sync(MASK_ALL, *value, delta);
  return *reinterpret_cast<ScanState const*>(&result);
}

static __device__ __forceinline__ ScanState shfl_sync_status(ScanState const& state, int src_lane)
{
  auto const* src = reinterpret_cast<double const*>(&state);
  ScanState ret;
  auto* dst = reinterpret_cast<double*>(&ret);
  *dst = __shfl_sync(MASK_ALL, *src, src_lane);
  return ret;
}

template <int blockSize>
__device__ bool block_any_sync(bool value)
{
  int const warp_any = __any_sync(MASK_ALL, value);
  __shared__ uint sh_chunk;
  if (threadIdx.x == 0) sh_chunk = 0;
  __syncthreads();

  auto lane = threadIdx.x & (warpSize - 1);
  if (lane == 0)
    atomicAdd_block(&sh_chunk, warp_any);
  __syncthreads();
  return sh_chunk > 0;
}

// the correct status is stored in the last lane only!
__device__ ScanState warp_accumulate_prefix(ScanState thread_state)
{
  auto const lane = threadIdx.x & (warpSize - 1);
  for (auto s = 1; s < warpSize; s *= 2) {
    auto const tmp = shfl_up_sync_status(thread_state, s);
    if (lane >= s) {
      if (tmp.status == TileStatus::unavailable) {
        thread_state.status = tmp.status;
        thread_state.sum = 0;
      }
      else if (thread_state.status == TileStatus::local_prefix_available) {
        thread_state.sum += tmp.sum;
        thread_state.status = tmp.status;
      }
    }
  }
  return thread_state;
}

__device__ ScanState warp_accumulate_final_prefix_cub(ScanState thread_state)
{
  // set to zero all but the last lanes with global prefix
  auto const lane = threadIdx.x & (warpSize - 1);
  const auto final_mask = __ballot_sync(MASK_ALL, thread_state.status == TileStatus::global_prefix_available);
  const int last_lane_with_global_prefix = warpSize - 1 - __clz(final_mask);
  thread_state.sum = (lane < last_lane_with_global_prefix) ? 0.f : thread_state.sum;

  using WarpReduce = cub::WarpReduce<float>;
  __shared__ typename WarpReduce::TempStorage temp_storage;
  thread_state.sum = WarpReduce(temp_storage).Sum(thread_state.sum);
  thread_state.sum = __shfl_sync(MASK_ALL, thread_state.sum, 0);
  __syncwarp();

  return thread_state;
}

__device__ ScanState warp_accumulate_final_prefix(ScanState thread_state)
{
  // set to zero all but the last lanes with global prefix
  auto const lane = threadIdx.x & (warpSize - 1);
  const auto final_mask = __ballot_sync(MASK_ALL, thread_state.status == TileStatus::global_prefix_available);
  const int last_lane_with_global_prefix = warpSize - 1 - __clz(final_mask);
  thread_state.sum = (lane < last_lane_with_global_prefix) ? 0.f : thread_state.sum;

  for (auto s = 1; s < warpSize; s *= 2) {
    auto const tmp = __shfl_up_sync(MASK_ALL, thread_state.sum, s);
    if (lane >= s) {
      thread_state.sum += tmp;
    }
  }
  thread_state.sum = __shfl_sync(MASK_ALL, thread_state.sum, warpSize - 1);
  return thread_state;
}


__device__ float warp_lookback(uint32_t chunk, ScanState* states,
                               volatile float* sh_prefix_sum)
{
  auto const lane = threadIdx.x;
  int thread_chunk = chunk - (warpSize - lane);
  constexpr int last_lane = warpSize - 1;

  ScanState thread_state{TileStatus::global_prefix_available, 0.f};
  float prefix_sum{0.f};
  while (true) {
    thread_state = (thread_chunk >= 0) ? load_status(&states[thread_chunk]) :
                                      ScanState{TileStatus::global_prefix_available, 0.f};

    // check if there are unavailable prefixes
    auto const must_reload = __any_sync(MASK_ALL, thread_state.status == TileStatus::unavailable);
    if (must_reload) continue;

    prefix_sum += thread_state.sum;

    // check if any thread ran into finished state
    auto const any_finished = __any_sync(MASK_ALL, thread_state.status == TileStatus::global_prefix_available);
    if (any_finished) break;

    // keep looking back
    thread_chunk -= warpSize;
  }

  thread_state.sum = prefix_sum;
  // thread_state = warp_accumulate_final_prefix_cub(thread_state);
  thread_state = warp_accumulate_final_prefix(thread_state);

  if (lane == last_lane) {
    *sh_prefix_sum = thread_state.sum;
  }
  return thread_state.sum;
}

template <int blockSize>
__device__ void warp_lookback_preloaded(uint32_t chunk, ScanState* states,
                                        ScanState* _preloaded_states,
                                        volatile float* sh_prefix_sum)
{
  auto const lane = threadIdx.x;
  int thread_chunk = chunk - (warpSize - lane);
  int block_pos = blockSize - (warpSize - lane);
  constexpr int last_lane = warpSize - 1;

  ScanState thread_state{TileStatus::global_prefix_available, 0.f};
  float prefix_sum{0.f};
  while (true) {
    // first try to read from local memory
    thread_state = (block_pos >= 0) ? _preloaded_states[block_pos] :
                                      (thread_chunk >= 0) ? load_status(&states[thread_chunk]) :
                                                             ScanState{TileStatus::global_prefix_available, 0.f};
    // reload statefrom global memory
    while (__any_sync(MASK_ALL, thread_state.status == TileStatus::unavailable))
      thread_state = (thread_chunk >= 0) ? load_status(&states[thread_chunk]) :
                                           ScanState{TileStatus::global_prefix_available, 0.f};

    prefix_sum += thread_state.sum;

    // check if any thread ran into finished state
    auto const any_finished = __any_sync(MASK_ALL, thread_state.status == TileStatus::global_prefix_available);
    if (any_finished) break;

    // keep looking back
    thread_chunk -= warpSize;
    block_pos -= warpSize;
  }

  thread_state.sum = prefix_sum;
  // thread_state = warp_accumulate_final_prefix_cub(thread_state);
  thread_state = warp_accumulate_final_prefix(thread_state);

  if (lane == last_lane) {
    *sh_prefix_sum = thread_state.sum;
  }
}


template<int blockSize>
__device__ float block_accumulate_prefix(ScanState& thread_state)
{
  constexpr int shmem_size = blockSize/warpSize;
  __shared__ ScanState warp_states[shmem_size];
  __shared__ ScanState _ans;

  auto const tid = threadIdx.x;
  if (tid < shmem_size)
    warp_states[tid] = ScanState{TileStatus::local_prefix_available, 0.f};

  // local accumulate
  thread_state = warp_accumulate_prefix(thread_state);
  constexpr int last_lane = warpSize - 1;
  // thread_state = shfl_sync_status(thread_state, last_lane);
  auto const warp = tid / warpSize;
  auto const lane = tid & (warpSize - 1);
  if (lane == last_lane) {
    warp_states[warp] = thread_state;
  }
  __syncthreads();

  if (warp == 0) {
    auto warp_state = (lane < shmem_size) ? warp_states[lane] : ScanState{TileStatus::local_prefix_available, 0.f};
    warp_state = warp_accumulate_prefix(warp_state);
    if (lane == warpSize - 1) {
      _ans = warp_state;
    }

  }
  __syncthreads();

  return _ans.sum;
}

template<int blockSize>
__device__ float block_lookback(uint32_t chunk, ScanState* states)
{
  ScanState thread_state{TileStatus::global_prefix_available, 0.f};
  int thread_chunk = chunk - (blockSize - threadIdx.x);
  float prefix_sum{0.f};
  while (true) {
    // no need to block any sync right now
    // thread_state = (thread_chunk >= 0) ? load_status(&states[thread_chunk]) :
    //                                     ScanState{TileStatus::global_prefix_available, 0.f};
    // auto const must_reload = block_any_sync<blockSize>(thread_state.status == TileStatus::unavailable);
    // if (must_reload) continue;
    // prefix_sum += thread_state.sum;

    // let warps load until the data is available indenpendently
    do {
      thread_state = (thread_chunk >= 0) ? load_status(&states[thread_chunk]) :
          ScanState{TileStatus::global_prefix_available, 0.f};
    } while (__any_sync(MASK_ALL, thread_state.status == TileStatus::unavailable));
    prefix_sum += thread_state.sum;
    __syncthreads();

    // check if any thread ran into finished state
    auto const any_finished = block_any_sync<blockSize>(thread_state.status == TileStatus::global_prefix_available);
    if (any_finished) break;

    // keep looking back
    thread_chunk -= blockSize;
  }
  // get total prefix sum
  thread_state.sum = prefix_sum;
  return block_accumulate_prefix<blockSize>(thread_state);
}

template <int blockSize>
__global__ void scan_serial_lookback(float* out, float const* in, int n,
                                     ScanState* states, uint* tile_counter)
{
  __shared__ uint sh_chunk;

  auto tid = threadIdx.x;
  if (tid == 0) {
    auto chunk = atomicAdd(tile_counter, 1u);
    store_status(&states[chunk], ScanState{TileStatus::unavailable, 0.f});
    sh_chunk = chunk;
  }
  __syncthreads();

  auto const chunk = sh_chunk;
  auto const i = chunk*blockSize + tid;
  auto const thread_value = (i < n) ? in[i] : 0.f;
  auto const local_sum = block_scan<blockSize>(thread_value);
  bool const last_thread_in_block = (tid == blockSize - 1);

  if (last_thread_in_block) {
    store_status(&states[chunk], ScanState{TileStatus::local_prefix_available, local_sum});
  }

  auto prefix_sum = serial_lookback(chunk, states);
  auto aggregate = local_sum + prefix_sum;

  if (last_thread_in_block) {
    store_status(&states[chunk], ScanState{TileStatus::global_prefix_available, aggregate});
  }
  if (i < n) {
    out[i] = aggregate;
  }
}

template <int blockSize>
__global__ void scan_warp_lookback(float* out, float const* in, int n,
                                   ScanState* states, uint* tile_counter)
{
  __shared__ uint sh_chunk;
  __shared__ float sh_prefix_sum;

  // select chunk to work on
  auto tid = threadIdx.x;
  if (tid == 0) {
    auto chunk = atomicAdd(tile_counter, 1u);
    store_status(&states[chunk], ScanState{TileStatus::unavailable, 0.f});
    sh_chunk = chunk;
  }
  __syncthreads();

  // local prefix sum
  auto const chunk = sh_chunk;
  auto const i = chunk*blockSize + tid;
  auto const thread_value = (i < n) ? in[i] : 0.f;
  auto const local_sum = block_scan<blockSize>(thread_value);

  // update chunk status to local
  bool const last_thread_in_block = (tid == blockSize - 1);
  if (last_thread_in_block) {
    auto const status = (chunk != 0) ? TileStatus::local_prefix_available :
                                       TileStatus::global_prefix_available;
    store_status(&states[chunk], ScanState{status, local_sum});
  }

  // lookback
  if (threadIdx.x < warpSize)
    warp_lookback(chunk, states, &sh_prefix_sum);
  __syncthreads();

  // aggregate result
  float prefix_sum;
  if (tid & (warpSize - 1))
    prefix_sum = sh_prefix_sum;
  prefix_sum = __shfl_sync(MASK_ALL, sh_prefix_sum, warpSize - 1);

  auto const aggregate = local_sum + prefix_sum;

  // update chunk status
  if (last_thread_in_block) {
    store_status(&states[chunk], ScanState{TileStatus::global_prefix_available, aggregate});
  }
  // write output
  if (i < n) out[i] = aggregate;
}

template <int blockSize, int itemsPerThread>
__global__ void scan_cubbish_warp_lookback(float* out, float const* in, int n,
                                   ScanState* states, uint* tile_counter)
{
  __shared__ uint sh_chunk;
  __shared__ float sh_prefix_sum;

  // select chunk to work on
  auto tid = threadIdx.x;
  if (tid == 0) {
    auto chunk = atomicAdd(tile_counter, 1u);
    store_status(&states[chunk], ScanState{TileStatus::unavailable, 0.f});
    sh_chunk = chunk;
  }
  __syncthreads();

  using BlockLoadT = cub::BlockLoad<float, blockSize, itemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockStoreT = cub::BlockStore<float, blockSize, itemsPerThread, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  using BlockScanT = cub::BlockScan<float, blockSize>;

  __shared__ union TempStorage {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;
    typename BlockScanT::TempStorage scan;
  } temp_storage;

  auto const chunk = sh_chunk;
  float thread_data[itemsPerThread];
  BlockLoadT(temp_storage.load).Load(in + chunk*blockSize*itemsPerThread, thread_data, n);
  __syncthreads();

  float local_sum;
  BlockScanT(temp_storage.scan).InclusiveSum(thread_data, thread_data, local_sum);
  // __syncthreads(); // no need to sync

  // update chunk status to local
  bool const last_thread_in_block = (tid == blockSize - 1);
  if (last_thread_in_block) {
    auto const status = (chunk != 0) ? TileStatus::local_prefix_available :
                                       TileStatus::global_prefix_available;
    store_status(&states[chunk], ScanState{status, local_sum});
  }

  // lookback
  if (threadIdx.x < warpSize)
    warp_lookback(chunk, states, &sh_prefix_sum);
  __syncthreads();

  // aggregate result
  // float prefix_sum;
  // if (tid & (warpSize - 1))
  //   prefix_sum = sh_prefix_sum;
  // prefix_sum = __shfl_sync(MASK_ALL, sh_prefix_sum, warpSize - 1);
  float prefix_sum = sh_prefix_sum;

  // update chunk status
  if (last_thread_in_block) {
    store_status(&states[chunk], ScanState{TileStatus::global_prefix_available, prefix_sum + local_sum});
  }

  for (int i = 0; i < itemsPerThread; i++) {
    thread_data[i] += prefix_sum;
  }

  BlockStoreT(temp_storage.store).Store(out + chunk*blockSize*itemsPerThread, thread_data, n);
}


template <int blockSize>
__global__ void scan_warp_lookback_blockload(float* out, float const* in, int n,
                                             ScanState* states, uint* tile_counter)
{
  __shared__ uint sh_chunk;
  __shared__ float sh_prefix_sum;
  __shared__ ScanState _block_state[blockSize];

  // select chunk to work on
  auto tid = threadIdx.x;
  if (tid == 0) {
    auto chunk = atomicAdd(tile_counter, 1u);
    store_status(&states[chunk], ScanState{TileStatus::unavailable, 0.f});
    sh_chunk = chunk;
  }
  __syncthreads();

  // local prefix sum
  auto const chunk = sh_chunk;
  auto const i = chunk*blockSize + tid;
  auto const thread_value = (i < n) ? in[i] : 0.f;
  auto const local_sum = block_scan<blockSize>(thread_value);

  // update chunk status to local
  bool const last_thread_in_block = (tid == blockSize - 1);
  if (last_thread_in_block) {
    auto const status = (chunk != 0) ? TileStatus::local_prefix_available :
                                       TileStatus::global_prefix_available;
    store_status(&states[chunk], ScanState{status, local_sum});
  }

  // block load the data
  {
    int thread_chunk = chunk - (blockSize - threadIdx.x);
    _block_state[tid]  = (thread_chunk >= 0) ? load_status(&states[thread_chunk]) :
                                               ScanState{TileStatus::global_prefix_available, 0.f};
  }
  __syncthreads();

  // lookback
  if (threadIdx.x < warpSize)
    warp_lookback_preloaded<blockSize>(chunk, states, _block_state, &sh_prefix_sum);
  __syncthreads();

  // aggregate result
  float const prefix_sum = sh_prefix_sum;
  auto const aggregate = local_sum + prefix_sum;

  // update chunk status
  if (last_thread_in_block) {
    store_status(&states[chunk], ScanState{TileStatus::global_prefix_available, aggregate});
  }
  // write output
  if (i < n) out[i] = aggregate;
}


template <int blockSize, int itemsPerThread>
__global__ void scan_block_lookback(float* out, float const* in, int n,
                                    ScanState* states, uint* tile_counter)
{
  auto tid = threadIdx.x;
  // select chunk to work on
  __shared__ uint sh_chunk;
  if (tid == 0) {
    auto chunk = atomicAdd(tile_counter, 1u);
    store_status(&states[chunk], ScanState{TileStatus::unavailable, 0.f});
    sh_chunk = chunk;
  }
  __syncthreads();
  auto const chunk = sh_chunk;

  constexpr int chunkSize = blockSize*itemsPerThread;
  __shared__ float _data[chunkSize];
  float thread_data[itemsPerThread];
  block_load<blockSize, itemsPerThread>(chunk,in, n, thread_data, _data);
  __syncthreads();

   block_scan<blockSize, itemsPerThread>(thread_data, _data);

  // update chunk status to local
  bool const last_thread_in_block = (tid == blockSize - 1);
  if (last_thread_in_block) {
    auto const status = (chunk != 0) ? TileStatus::local_prefix_available :
                                       TileStatus::global_prefix_available;
    store_status(&states[chunk], ScanState{status, thread_data[itemsPerThread-1]});
  }

  // lookback
  auto const prefix_sum = block_lookback<blockSize>(chunk, states);
  // aggregate result
  for (int i = 0; i < itemsPerThread; i++) {
    thread_data[i] += prefix_sum;
  }

  // update chunk status
  if (last_thread_in_block) {
    store_status(&states[chunk], ScanState{TileStatus::global_prefix_available,
        thread_data[itemsPerThread-1]});
  }

  block_store<blockSize, itemsPerThread>(chunk, out, n, thread_data, _data);
}
// template <int blockSize>
// __global__ void scan_block_lookback(float* out, float const* in, int n,
//                                     ScanState* states, uint* tile_counter)
// {
//   auto tid = threadIdx.x;
//   // select chunk to work on
//   __shared__ uint sh_chunk;
//   if (tid == 0) {
//     auto chunk = atomicAdd(tile_counter, 1u);
//     store_status(&states[chunk], ScanState{TileStatus::unavailable, 0.f});
//     sh_chunk = chunk;
//   }
//   __syncthreads();

//   // local prefix sum
//   auto const chunk = sh_chunk;
//   auto const i = chunk*blockSize + tid;
//   auto const thread_value = (i < n) ? in[i] : 0.f;
//   auto const local_sum = block_scan<blockSize>(thread_value);

//   // update chunk status to local
//   bool const last_thread_in_block = (tid == blockSize - 1);
//   if (last_thread_in_block) {
//     auto const status = (chunk != 0) ? TileStatus::local_prefix_available :
//                                        TileStatus::global_prefix_available;
//     store_status(&states[chunk], ScanState{status, local_sum});
//   }

//   // lookback
//   auto const prefix_sum = block_lookback<blockSize>(chunk, states);
//   // aggregate result
//   auto const aggregate = local_sum + prefix_sum;

//   // update chunk status
//   if (last_thread_in_block) {
//     store_status(&states[chunk], ScanState{TileStatus::global_prefix_available, aggregate});
//   }
//   // write output
//   if (i < n) out[i] = aggregate;
// }

template <int blockSize, int itemsPerThread>
__global__ void fancy_scan_lookback(float* __restrict__ out,
                                    float const* __restrict__ in, int n,
                                    ScanState* __restrict__ states, uint* tile_counter)
{
  constexpr int chunkSize = blockSize*itemsPerThread;
  __shared__ float _data[chunkSize];
  auto const tid = threadIdx.x;

  // fetch chunk
  __shared__ uint sh_chunk;
  __shared__ float sh_prefix_sum;
  if (tid == 0) {
    auto chunk = atomicAdd(tile_counter, 1u);
    store_status(&states[chunk], ScanState{TileStatus::unavailable, 0.f});
    sh_chunk = chunk;
  }
  __syncthreads();
  auto const chunk = sh_chunk;

  // auto lane = threadIdx.x & (warpSize-1);
  // auto warp = tid / warpSize;

  // // load data
  // for (int i = 0; i < itemsPerThread; i++) {
  //   int pos_glob = chunk*chunkSize + i*blockSize + tid;
  //   int pos = i*blockSize + tid;
  //   _data[pos] = (pos_glob < n) ? in[pos_glob] : 0.f;
  // }
  // __syncthreads();

  // // load into thread local
  // float thread_data[itemsPerThread];
  // for (int i = 0; i < itemsPerThread; i++) {
  //   auto idx = warp*warpSize*itemsPerThread + lane + i*warpSize;
  //   thread_data[i] = _data[idx];
  // }

  float thread_data[itemsPerThread];
  block_load<blockSize, itemsPerThread>(chunk, in, n, thread_data, _data);
  __syncthreads();

  /* Scan */
   block_scan<blockSize, itemsPerThread>(thread_data, _data);
   // __syncthreads(); // no need to sync here

  // update chunk status to local
  bool const last_thread_in_block = (tid == blockSize - 1);
  float local_sum;
  if (last_thread_in_block) {
    auto const status = (chunk != 0) ? TileStatus::local_prefix_available :
                                       TileStatus::global_prefix_available;
    local_sum = thread_data[itemsPerThread-1];
    store_status(&states[chunk], ScanState{status, local_sum});
  }

  if (threadIdx.x < warpSize) {
    warp_lookback(chunk, states, &sh_prefix_sum);
  }
  __syncthreads();

  // aggregate result
  float prefix_sum = sh_prefix_sum;
  for (int i = 0; i < itemsPerThread; i++) {
    thread_data[i] += prefix_sum;
  }
  if (last_thread_in_block) {
    store_status(&states[chunk], ScanState{TileStatus::global_prefix_available, thread_data[itemsPerThread-1]});
  }

  block_store<blockSize, itemsPerThread>(chunk, out, n, thread_data, _data);
}

void __noinline__ scan_single_pass(thrust::device_vector<float> & in,
                                   thrust::device_vector<float> & out,
                                   int blockSize,
                                   auto && launch_kernel,
                                   int items_per_thread = 1)
{
  uint32_t* tile_counter;
  cudaMalloc(&tile_counter, sizeof(uint32_t));
  cudaMemset(tile_counter, 0, sizeof(unsigned int));

  int chunkSize = blockSize * items_per_thread;
  int64_t const numBlocks = (in.size() + chunkSize - 1) / chunkSize;

  ScanState *states;
  cudaMalloc(&states, numBlocks*sizeof(ScanState));

  // thrust::device_vector<ScanState> states_vector(numBlocks);
  // auto* states = states_vector.data().get();

  launch_kernel<<<numBlocks, blockSize>>>
      (out.data().get(), in.data().get(), in.size(), states, tile_counter);

  cudaFree(states);
  cudaFree(tile_counter);
}
