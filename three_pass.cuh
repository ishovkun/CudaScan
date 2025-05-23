// __global__ void add_partial_sums(float* out, float* partial_sums, int n) {
//   auto blockOffset = blockIdx.x * blockDim.x;
//   auto tid = threadIdx.x;
//   if (blockIdx.x > 0 && blockOffset + tid < n) {
//     out[blockOffset + tid] += partial_sums[blockIdx.x - 1];
//   }
// }
__global__ void add_partial_sums(float* out, float* partial_sums, int n) {
  auto blockOffset = blockIdx.x * blockDim.x;
  auto tid = threadIdx.x;
  __shared__ float partial_sum;
  if (tid == 0) {
    partial_sum = (blockIdx.x > 0) ? partial_sums[blockIdx.x - 1] : 0.f;
  }
  __syncthreads();
  if (blockOffset + tid < n) {
    out[blockOffset + tid] += partial_sum;
  }
}

template <int chunksPerBlock>
__global__ void addPartialSums(float* out, float* chunkPrefixes, int chunkSize, int n) {
  auto tid = threadIdx.x;
  auto firstChunk = blockIdx.x*chunksPerBlock;
  auto numChunks = n / chunkSize;
  __shared__ float cache[chunksPerBlock];

  for (int i = 0; blockDim.x*i + tid < chunksPerBlock; i++) {
    auto chunk = blockIdx.x*chunksPerBlock + blockDim.x*i + tid - 1;
    auto chunkLocal = blockDim.x*i + tid;
    if (chunkLocal < chunksPerBlock)
      cache[chunkLocal] = (chunk < numChunks) ? chunkPrefixes[chunk] : 0.f;
  }
  __syncthreads();

  for (int idx = blockIdx.x*chunksPerBlock*chunkSize + tid; idx < (blockIdx.x+1)*chunksPerBlock*chunkSize; idx += blockDim.x) {
    auto chunkLocal = idx / chunkSize - firstChunk;
    if (idx < n) {
      out[idx] += cache[chunkLocal];
    }
  }
}

template <int blockSize>
__global__ void scan_double_buffer(float* out, float* in, int n, float* partial_sums) {
  auto tid = threadIdx.x;
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float shared[2][blockSize];
  if (i < n) shared[0][tid] = in[i];
  else       shared[0][tid] = 0.f;
  __syncthreads();

  // Double buffering, the table prints whatever in the output buffer
  // idx  | 0  | 1      | 2       | 3      | 4      | 5      | 6      | 7      | 8      | 9       | 10       | 11      | 12      | 13      | 14       | 15       |
  // | init | x0 | x1     | x2      | x3     | x4     | x5     | x6     | x7     | x8     | x9      | x10      | x11     | x12     | x13     | x14      | x15      |
  // | s=1  | x0 | x0,x1  | x1,x2   | x2,x3  | x3,x4  | x4, x5 | x5,x6  | x6,x7  | x7,x8  | x8,x9   | x9,x10   | x10,x11 | x11,x12 | x12,x13 | x14      | x14,x15  |
  // | s=2  | x0 | x0..x1 | x0...x2 | x0..x3 | x1..x4 | x2..x5 | x3..x6 | x4..x7 | x5..x8 | x6...x9 | x7...x10 | x7..x11 | x8..x12 | x9..x13 | x10..x14 | x11..x15 |
  // | s=3  | x0 | x0..x1 | x0...x2 | x0..x3 | x0..x4 | x0..x5 | x0..x6 | x0..x7 | x1..x8 | x2...x9 | x3...x10 | x4..x11 | x5..x12 | x6..x13 | x7..x14  | x8..x15  |
  int rbuf = 0; // read buffer idx
  for (int s = 1; s < blockSize; s *= 2) {
    float add = (tid >= s) ? shared[rbuf][tid-s] : 0.f;
    shared[!rbuf][tid] = shared[rbuf][tid] + add;
    rbuf = !rbuf;
    __syncthreads();
  }
  out[i] = shared[rbuf][tid];
  if (tid == 0) partial_sums[blockIdx.x] = shared[rbuf][blockSize-1];
}

template <int blockSize>
__global__ void scan_single_buffer(float* out, float const* in, int n, float* partial_sums) {
  auto tid = threadIdx.x;
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sh[blockSize];
  if (i < n) sh[tid] = in[i];
  else       sh[tid] = 0.f;
  __syncthreads();

  // upsweep
  /*
      | idx  | 0 | 1 | 2 |  3 | 4 |  5 | 6 |  7 | 8 |  9 | 10 | 11 | 12 | 13 | 14 |  15 |
      | init | 1 | 2 | 3 |  4 | 5 |  6 | 7 |  8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |  16 |
              \---|   \----|    \--|    \----|   \---|    \-----|    \----|   \-----|
      | s=1  |   | 3 |   |  7 |   | 11 |   | 15 |   | 19 |    | 23 |    | 27 |    |  31 |
                  \--------|        \--------|        \---------|        \----------|
      | s=2  |   |   |   | 10 |   |    |   | 26 |   |    |    | 42 |    |    |    |  58 |
                            \----------------|                   \-------------------|
      | s=4  |   |   |   |    |   |    |   | 36 |   |    |    |    |    |    |    | 100 |
                                              \--------------------------------------|
      | s=8  |   |   |   |    |   |    |   |    |   |    |    |    |    |    |    | 136 |
  */
  for (int s = 1; s < blockSize; s *= 2) {
    if ((tid+1) % (2*s) == 0) {
      sh[tid] += sh[tid-s];
    }
    __syncthreads();
  }
  if (tid == blockSize - 1) {
    partial_sums[blockIdx.x] = sh[tid];
  }
  /*
    downsweep
    | idx  | 0 | 1 | 2 |  3 |  4 |  5 |  6 | 7  |  8 |  9 | 10 | 11 | 12 | 13 | 14 |  15 |
    | init | 1 | 3 | 3 | 10 |  5 | 11 |  7 | 36 |  9 | 19 | 11 | 42 | 13 | 27 | 15 | 136 |
                                            \--------->>--------\
    | s=4  |   |   |   |    |    |    |    | X  |    |    |    | 78 |    |    |    |     |
                          \--->>----\       \--->>----\          \--->>---\
    | s=2  |   |   |   |  X |    | 21 |    | X  |    | 55 |    |  X |    | 69 |    |     |
                \->-\    \->-\     \->-\    \->-\      \->-\     \->-\     \->-\
    | s=1  |   | X | 6 |  X | 15 |  X | 28 | X  | 45 |  X | 66 |  X | 91 |  X | 85 |     |
   */
  for (int s = blockSize/2; s > 0; s /= 2) {
    if ((tid+1) % (2*s) == 0) {
      sh[tid+s] += sh[tid];
    }
    __syncthreads();
  }
  if (i < n) out[i] = sh[tid];
}

template <int blockSize, int itemsPerThread>
__global__ void scan_work_efficient(float* out, float const* in,
                                    int input_size,
                                    float* partial_sums) {
  constexpr int chunkSize = blockSize*itemsPerThread;
  __shared__ float shared[chunkSize];
  auto const tid = threadIdx.x;
  auto const chunk = blockIdx.x;

  block_load<blockSize,itemsPerThread>(chunk, in, input_size, shared);
  __syncthreads();

  // upsweep
  // the rest is the same as in the previous function, apart from
  // the s limit: in the previous function it was [1, blockSize/2],
  // now it's [1, blockSize]
  for (int s = 1; s <= chunkSize; s *= 2) {
    auto idx = 2*s*tid + s - 1;
    if (idx+s < chunkSize) {
      shared[idx+s] += shared[idx];
    }
    __syncthreads();
  }
  for (int s = chunkSize; s > 0; s /= 2) {
    auto idx = 2*s*(tid+1) - 1;
    if (idx + s < chunkSize) {
      shared[idx+s] += shared[idx];
    }
    __syncthreads();
  }

  block_store<blockSize, itemsPerThread>(chunk, out, input_size, shared);
  if (tid == 0) {
    partial_sums[chunk] = shared[chunkSize-1];
  }
}
// template <int blockSize>
// __global__ void scan_work_efficient(float* out, float const* in, int n, float* partial_sums) {
//   __shared__ float sh[2*blockSize];

//   auto tid = threadIdx.x;
//   auto ia = blockIdx.x * 2* blockSize + threadIdx.x;
//   auto ib = blockIdx.x * 2* blockSize + threadIdx.x + blockSize;
//   if (ia < n) sh[tid] = in[ia];
//   else sh[tid] = 0;
//   if (ib) sh[tid+blockSize] = in[ib];
//   else sh[tid+blockSize] = 0;
//   __syncthreads();

//   // upsweep
//   // the rest is the same as in the previous function, apart from
//   // the s limit: in the previous function it was [1, blockSize/2],
//   // now it's [1, blockSize]
//   for (int s = 1; s <= blockSize; s *= 2) {
//     auto idx = 2*s*tid + s - 1;
//     if (idx+s < 2*blockSize) {
//       sh[idx+s] += sh[idx];
//     }
//     __syncthreads();
//   }
//   if (tid == 0) {
//     partial_sums[blockIdx.x] = sh[2*blockSize-1];
//   }
//   // no need to __syncthreads(); (we don't touch the last element)
//   for (int s = blockSize; s > 0; s /= 2) {
//     auto idx = 2*s*(tid+1) - 1;
//     if (idx + s < 2*blockSize) {
//       sh[idx+s] += sh[idx];
//     }
//     __syncthreads();
//   }
//   if (ia < n) {
//     out[ia] = sh[tid];
//   }
//   if (ib < n) {
//     out[ib] = sh[tid+blockSize];
//   }
// }

// template <int blockSize>
// __global__ void scan_on_registers(float* out, float const* in, int n, float* partial_sums) {
//   // extra element to avoid ifs for thread 0
//   constexpr int shmem_size = 1 + blockSize/warpSize;
//   __shared__ float warp_sums[shmem_size];
//   // fill shared sums with zero
//   // we should really only set warp_sums[0] = 0
//   auto tid = threadIdx.x;
//   if (tid < shmem_size)
//     warp_sums[tid] = 0.f;

//   auto i = blockIdx.x*blockDim.x + threadIdx.x;
//   auto thread_value = (i < n) ? in[i] : 0;

//   auto thread_prefix_within_warp = warp_scan(thread_value);
//   __syncthreads();

//   auto warp = threadIdx.x / warpSize;
//   auto lane = threadIdx.x & (warpSize-1);
//   if (lane == warpSize-1)
//     warp_sums[1+warp] = thread_prefix_within_warp;
//   __syncthreads();

//   // scan warp sums in a separate warp
//   // max block size = 1024; 1024/32 = 32 -> should fit inside a warp
//   if (warp == 0)
//     warp_sums[1+lane] = warp_scan(warp_sums[1+lane]);
//   __syncthreads();

//   if (i < n) {
//     out[i] = thread_prefix_within_warp + warp_sums[warp];
//   }
//   if (tid == 0) {
//     partial_sums[blockIdx.x] = warp_sums[shmem_size - 1];
//   }
// }

// template <int blockSize>
// __global__ void scan_cub(float* out, float const* in, int n, float* partial_sums) {

//   using BlockScanT = cub::BlockScan<float, blockSize>;
//   __shared__ typename BlockScanT::TempStorage temp_storage;

//   auto i = blockIdx.x*blockDim.x + threadIdx.x;
//   auto thread_value = (i < n) ? in[i] : 0;

//   BlockScanT(temp_storage).InclusiveSum(thread_value, thread_value, partial_sums[blockIdx.x]);
//   if (i < n)
//     out[i] = thread_value;

//   if (threadIdx.x == blockSize-1) {
//     partial_sums[blockIdx.x] = thread_value;
//   }
// }

template <int blockSize, int itemsPerThread>
__global__ void scan_cub(float* out, float const* in, int n, float* partial_sums) {

  using BlockLoadT = cub::BlockLoad<float, blockSize, itemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockStoreT = cub::BlockStore<float, blockSize, itemsPerThread, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  using BlockScanT = cub::BlockScan<float, blockSize>;

  __shared__ union TempStorage {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;
    typename BlockScanT::TempStorage scan;
  } temp_storage;

  float thread_data[itemsPerThread];
  BlockLoadT(temp_storage.load).Load(in + blockIdx.x*blockSize*itemsPerThread, thread_data, n);
  __syncthreads();

  float aggregate;
  BlockScanT(temp_storage.scan).InclusiveSum(thread_data, thread_data, aggregate);
  __syncthreads();

  BlockStoreT(temp_storage.store).Store(out + blockIdx.x*blockSize*itemsPerThread, thread_data, n);

  if (threadIdx.x == blockSize-1) {
    partial_sums[blockIdx.x] = aggregate;
  }
}

template <int blockSize, int itemsPerThread>
__global__ void scan_on_registers(float* out, float const* in, int n, float* partial_sums) {
  constexpr int chunkSize = blockSize*itemsPerThread;
  __shared__ float _data[chunkSize];
  int tid = threadIdx.x;
  auto const chunk = blockIdx.x;

  float thread_data[itemsPerThread];
  block_load<blockSize, itemsPerThread>(chunk,in, n, thread_data, _data);
  __syncthreads(); // need to reuse _data

  /* Scan */
   block_scan<blockSize, itemsPerThread>(thread_data, _data);
   __syncthreads();

   block_store<blockSize, itemsPerThread>(chunk, out, n, thread_data, _data);

   if (tid == 0) {
     partial_sums[blockIdx.x] = _data[chunkSize-1];
   }
}


__host__ __device__ constexpr __forceinline__ int conflict_free_offset(int n) {
  return n + (n >> LOG_NUM_BANKS) + (n >> (2*LOG_NUM_BANKS));
}

template <int blockSize, int itemsPerThread>
__global__ void scan_conflict_free_padding(float* out, float const* in,
                                           int input_size, float* partial_sums)
{
  constexpr int chunkSize = blockSize*itemsPerThread;
  constexpr int shmem_size =  conflict_free_offset(chunkSize);
  __shared__ float shared[shmem_size];
  auto tid = threadIdx.x;
  auto const chunk = blockIdx.x;

  for (int i = 0; i < itemsPerThread; i++) {
    int global_id = chunk*chunkSize + i*blockSize + tid;
    int local_id = i*blockSize + tid;
    shared[conflict_free_offset(local_id)] = (global_id < input_size) ? in[global_id] : 0.f;
  }
  __syncthreads();

  // upsweep
  for (int s = 1; s <= chunkSize; s *= 2) {
    auto idx = 2*s*tid + s - 1;
    if (idx+s < chunkSize) {
      shared[conflict_free_offset(idx+s)] += shared[conflict_free_offset(idx)];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_sums[chunk] = shared[conflict_free_offset(chunkSize-1)];
  }

  // downsweep
  for (int s = chunkSize; s > 0; s /= 2) {
    auto idx = 2*s*(tid+1) - 1;
    if (idx + s < chunkSize) {
      shared[conflict_free_offset(idx+s)] += shared[conflict_free_offset(idx)];
    }
    __syncthreads();
  }

  for (int i = 0; i < itemsPerThread; i++) {
    int pos_glob = chunk*chunkSize + i*blockSize + tid;
    int pos = i*blockSize + tid;
    if (pos_glob < input_size) {
      out[pos_glob] = shared[conflict_free_offset(pos)];
    }
  }
}

// template <int blockSize>
// __global__ void scan_conflict_free_padding(float* out, float const* in, int n, float* partial_sums) {
//   constexpr int shmem_size =  conflict_free_offset(2*blockSize);
//   __shared__ float sh[shmem_size];
//   auto tid = threadIdx.x;
//   auto i = blockIdx.x * blockDim.x + threadIdx.x;

//   auto a_base = conflict_free_offset(2*tid);
//   auto b_base = conflict_free_offset(2*tid+1);

//   if (2*i < n) sh[a_base] = in[2*i];
//   else sh[a_base] = 0;
//   if (2*i+1 < n) sh[b_base] = in[2*i+1];
//   else sh[b_base] = 0;
//   __syncthreads();

//   // upsweep
//   for (int s = 1; s <= blockSize; s *= 2) {
//     auto idx = 2*s*tid + s - 1;
//     if (idx+s < 2*blockSize) {
//       sh[conflict_free_offset(idx+s)] += sh[conflict_free_offset(idx)];
//     }
//     __syncthreads();
//   }

//   if (tid == 0) {
//     partial_sums[blockIdx.x] = sh[conflict_free_offset(2*blockSize-1)];
//   }

//   // downsweep
//   for (int s = blockSize; s > 0; s /= 2) {
//     auto idx = 2*s*(tid+1) - 1;
//     if (idx + s < 2*blockSize) {
//       sh[conflict_free_offset(idx+s)] += sh[conflict_free_offset(idx)];
//     }
//     __syncthreads();
//   }
//   // write results
//   if (2*i < n) {
//     out[2*i] = sh[a_base];
//   }
//   if (2*i+1 < n) {
//     out[2*i+1] = sh[b_base];
//   }
// }

__device__ constexpr __forceinline__ int swizzle(int i) {
  return i ^ (i >> 5);
}

template <int blockSize, int itemsPerThread>
__global__ void scan_conflict_free_swizzle(float* out, float const* in,
                                           int input_size, float* partial_sums) {
  constexpr int chunkSize = blockSize*itemsPerThread;
  __shared__ float shared[chunkSize];
  auto const tid = threadIdx.x;
  auto const chunk = blockIdx.x;

  for (int i = 0; i < itemsPerThread; i++) {
    int global_id = chunk*chunkSize + i*blockSize + tid;
    int local_id = i*blockSize + tid;
    shared[swizzle(local_id)] = (global_id < input_size) ? in[global_id] : 0.f;
  }
  __syncthreads();

  // upsweep
  for (int s = 1; s <= blockSize; s *= 2) {
    auto idx = 2*s*tid + s - 1;
    if (idx+s < 2*blockSize) {
      shared[swizzle(idx+s)] += shared[swizzle(idx)];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_sums[blockIdx.x] = shared[swizzle(2*blockSize-1)];
  }

  // downsweep
  for (int s = blockSize; s > 0; s /= 2) {
    auto idx = 2*s*(tid+1) - 1;
    if (idx + s < 2*blockSize) {
      shared[swizzle(idx+s)] += shared[swizzle(idx)];
    }
    __syncthreads();
  }

  for (int i = 0; i < itemsPerThread; i++) {
    int pos_glob = chunk*chunkSize + i*blockSize + tid;
    int pos = i*blockSize + tid;
    if (pos_glob < input_size) {
      out[pos_glob] = shared[swizzle(pos)];
    }
  }
}

// template <int blockSize>
// __global__ void scan_conflict_free_swizzle(float* out, float const* in, int n, float* partial_sums) {
//   __shared__ float sh[2*blockSize];
//   auto tid = threadIdx.x;

//   auto a_base = swizzle(tid);
//   auto b_base = swizzle(blockSize + tid);
//   auto a = 2*blockIdx.x*blockDim.x + tid;
//   auto b = 2*blockIdx.x*blockDim.x + blockDim.x + tid;
//   if (a < n) sh[a_base] = in[a];
//   else sh[a_base] = 0;
//   if (b < n) sh[b_base] = in[b];
//   else sh[b_base] = 0;

//   // auto i = blockIdx.x * blockDim.x + threadIdx.x;
//   // auto a_base = swizzle(2*tid);
//   // auto b_base = swizzle(2*tid+1);
//   // if (2*i < n) sh[a_base] = in[2*i];
//   // else sh[a_base] = 0;
//   // if (2*i+1 < n) sh[b_base] = in[2*i+1];
//   // else sh[b_base] = 0;
//   __syncthreads();

//   // upsweep
//   for (int s = 1; s <= blockSize; s *= 2) {
//     auto idx = 2*s*tid + s - 1;
//     if (idx+s < 2*blockSize) {
//       sh[swizzle(idx+s)] += sh[swizzle(idx)];
//     }
//     __syncthreads();
//   }

//   if (tid == 0) {
//     partial_sums[blockIdx.x] = sh[swizzle(2*blockSize-1)];
//   }

//   // downsweep
//   for (int s = blockSize; s > 0; s /= 2) {
//     auto idx = 2*s*(tid+1) - 1;
//     if (idx + s < 2*blockSize) {
//       sh[swizzle(idx+s)] += sh[swizzle(idx)];
//     }
//     __syncthreads();
//   }
//   // write results
//   if (a < n) {
//     out[a] = sh[a_base];
//   }
//   if (b < n) {
//     out[b] = sh[b_base];
//   }
// }

void cub_device_scan(thrust::device_vector<float> & in, thrust::device_vector<float> & out)
{
  int num_items = in.size();
  float *d_in = thrust::raw_pointer_cast(in.data());
  float *d_out = thrust::raw_pointer_cast(out.data());
  // Determine temporary device storage requirements
  void     *d_temp_storage = nullptr;
  size_t   temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes,
      d_in, d_out, num_items);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run exclusive prefix sum
  cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes,
      d_in, d_out, num_items);
}


void __noinline__ scan(thrust::device_vector<float> & in,
                       thrust::device_vector<float> & out,
                       int block_size, int chunkSize, auto && scan_kernel) {
  auto problem_size = in.size();
  int num_blocks = (problem_size + chunkSize - 1) / chunkSize;

  thrust::device_vector<float> partial_sums(num_blocks, 0);

  scan_kernel<<<num_blocks, block_size>>>(thrust::raw_pointer_cast(out.data()),
                                          thrust::raw_pointer_cast(in.data()),
                                          problem_size,
                                          thrust::raw_pointer_cast(partial_sums.data()));
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  if (num_blocks > 1) {

    thrust::device_vector<float> partial_sums_of_partial_sums(num_blocks, 0);
    scan(partial_sums, partial_sums_of_partial_sums, block_size, chunkSize, scan_kernel);

    // we make a separate grid for accumulating partial sums
    {
      constexpr int chunksPerBlock = 64;
      auto nBlocksAddSums = (num_blocks + chunksPerBlock - 1) / chunksPerBlock;
      addPartialSums<chunksPerBlock><<<nBlocksAddSums, chunksPerBlock>>>(
          out.data().get(), partial_sums_of_partial_sums.data().get(),
          chunkSize, problem_size);
    }

    // add_partial_sums<<<num_blocks, chunkSize>>>(thrust::raw_pointer_cast(out.data()),
    //                                             thrust::raw_pointer_cast(partial_sums_of_partial_sums.data()),
    //                                             problem_size);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }
}
