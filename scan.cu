#include <chrono>
#include <iostream>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/iterator/zip_iterator.h>
#include "decoupled_lookback.cuh"
#include "common.cuh"
#include <iomanip>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    std::cout << "GPU assert failed" << std::endl;
    if (abort) exit(code);
  }
}

__global__ void add_partial_sums(float* out, float* partial_sums, int n) {
  auto blockOffset = blockIdx.x * blockDim.x;
  auto tid = threadIdx.x;
  if (blockIdx.x > 0 && blockOffset + tid < n) {
    out[blockOffset + tid] += partial_sums[blockIdx.x - 1];
  }
}

template <int blockSize>
__global__ void scan_naive(float* out, float* in, int n, float* partial_sums) {
  auto tid = threadIdx.x;
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sh[blockSize];
  if (i < n) sh[tid] = in[i];
  else       sh[tid] = 0.f;
  __syncthreads();

  // Double buffering, the table prints whatever in the output buffer
  // idx  | 0  | 1      | 2       | 3      | 4      | 5      | 6      | 7      | 8      | 9       | 10       | 11      | 12      | 13      | 14       | 15       |
  // | init | x0 | x1     | x2      | x3     | x4     | x5     | x6     | x7     | x8     | x9      | x10      | x11     | x12     | x13     | x14      | x15      |
  // | s=1  | x0 | x0,x1  | x1,x2   | x2,x3  | x3,x4  | x4, x5 | x5,x6  | x6,x7  | x7,x8  | x8,x9   | x9,x10   | x10,x11 | x11,x12 | x12,x13 | x14      | x14,x15  |
  // | s=2  | x0 | x0..x1 | x0...x2 | x0..x3 | x1..x4 | x2..x5 | x3..x6 | x4..x7 | x5..x8 | x6...x9 | x7...x10 | x7..x11 | x8..x12 | x9..x13 | x10..x14 | x11..x15 |
  // | s=3  | x0 | x0..x1 | x0...x2 | x0..x3 | x0..x4 | x0..x5 | x0..x6 | x0..x7 | x1..x8 | x2...x9 | x3...x10 | x4..x11 | x5..x12 | x6..x13 | x7..x14  | x8..x15  |
  int fin = 0; // flag, !fin = out
  for (int s = 1; s < blockSize; s *= 2) {
    if (tid >= s) {
      sh[(!fin)*blockSize + tid] = sh[fin*blockSize + tid - s] + sh[fin*blockSize + tid];
    } else {
      sh[(!fin) * blockSize + tid] = sh[fin * blockSize + tid];
    }

    fin = !fin;
    __syncthreads();
  }
  out[i] = sh[(fin)*blockSize + tid];
  if (tid == 0) partial_sums[blockIdx.x] = sh[fin*blockSize + blockSize-1];
}

template <int blockSize>
__global__ void scan_more_efficient(float* out, float const* in, int n, float* partial_sums) {
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

template <int blockSize>
__global__ void scan_work_efficient(float* out, float const* in, int n, float* partial_sums) {
  __shared__ float sh[2*blockSize];

  auto tid = threadIdx.x;
  auto ia = blockIdx.x * 2* blockSize + threadIdx.x;
  auto ib = blockIdx.x * 2* blockSize + threadIdx.x + blockSize;
  if (ia < n) sh[tid] = in[ia];
  else sh[tid] = 0;
  if (ib) sh[tid+blockSize] = in[ib];
  else sh[tid+blockSize] = 0;
  __syncthreads();

  // upsweep
  // the rest is the same as in the previous function, apart from
  // the s limit: in the previous function it was [1, blockSize/2],
  // now it's [1, blockSize]
  for (int s = 1; s <= blockSize; s *= 2) {
    auto idx = 2*s*tid + s - 1;
    if (idx+s < 2*blockSize) {
      sh[idx+s] += sh[idx];
    }
    __syncthreads();
  }
  if (tid == 0) {
    partial_sums[blockIdx.x] = sh[2*blockSize-1];
  }
  // no need to __syncthreads(); (we don't touch the last element)
  for (int s = blockSize; s > 0; s /= 2) {
    auto idx = 2*s*(tid+1) - 1;
    if (idx + s < 2*blockSize) {
      sh[idx+s] += sh[idx];
    }
    __syncthreads();
  }
  if (ia < n) {
    out[ia] = sh[tid];
  }
  if (ib < n) {
    out[ib] = sh[tid+blockSize];
  }
}

template <int blockSize>
__global__ void scan_on_registers(float* out, float const* in, int n, float* partial_sums) {
  // extra element to avoid ifs for thread 0
  constexpr int shmem_size = 1 + blockSize/WARP_SIZE;
  __shared__ float warp_sums[shmem_size];
  // fill shared sums with zero
  // we should really only set warp_sums[0] = 0
  auto tid = threadIdx.x;
  if (tid < shmem_size)
    warp_sums[tid] = 0.f;

  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto thread_value = (i < n) ? in[i] : 0;

  auto thread_prefix_within_warp = warp_scan(thread_value);
  __syncthreads();

  auto warp = threadIdx.x / WARP_SIZE;
  auto lane = threadIdx.x & (WARP_SIZE-1);
  if (lane == WARP_SIZE-1)
    warp_sums[1+warp] = thread_prefix_within_warp;
  __syncthreads();

  // scan warp sums in a separate warp
  // max block size = 1024; 1024/32 = 32 -> should fit inside a warp
  if (warp == 0)
    warp_sums[1+lane] = warp_scan(warp_sums[1+lane]);
  __syncthreads();

  if (i < n) {
    out[i] = thread_prefix_within_warp + warp_sums[warp];
  }
  if (tid == 0) {
    partial_sums[blockIdx.x] = warp_sums[shmem_size - 1];
  }
}



__host__ __device__ constexpr __forceinline__ int conflict_free_offset(int n) {
  return n + (n >> LOG_NUM_BANKS) + (n >> (2*LOG_NUM_BANKS));
}

template <int blockSize>
__global__ void scan_conflict_free(float* out, float const* in, int n, float* partial_sums) {
  constexpr int shmem_size =  conflict_free_offset(2*blockSize);
  __shared__ float sh[shmem_size];
  auto tid = threadIdx.x;
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  auto a_base = conflict_free_offset(2*tid);
  auto b_base = conflict_free_offset(2*tid+1);

  if (2*i < n) sh[a_base] = in[2*i];
  else sh[a_base] = 0;
  if (2*i+1 < n) sh[b_base] = in[2*i+1];
  else sh[b_base] = 0;
  __syncthreads();

  // upsweep
  for (int s = 1; s <= blockSize; s *= 2) {
    auto idx = 2*s*tid + s - 1;
    if (idx+s < 2*blockSize) {
      sh[conflict_free_offset(idx+s)] += sh[conflict_free_offset(idx)];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_sums[blockIdx.x] = sh[conflict_free_offset(2*blockSize-1)];
  }

  // downsweep
  for (int s = blockSize; s > 0; s /= 2) {
    auto idx = 2*s*(tid+1) - 1;
    if (idx + s < 2*blockSize) {
      sh[conflict_free_offset(idx+s)] += sh[conflict_free_offset(idx)];
    }
    __syncthreads();
  }
  // write results
  if (2*i < n) {
    out[2*i] = sh[a_base];
  }
  if (2*i+1 < n) {
    out[2*i+1] = sh[b_base];
  }
}

__device__ constexpr __forceinline__ int swizzle(int i) {
  return i ^ (i >> 5);
}

template <int blockSize>
__global__ void scan_conflict_free_swizzle(float* out, float const* in, int n, float* partial_sums) {
  __shared__ float sh[2*blockSize];
  auto tid = threadIdx.x;
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  auto a_base = swizzle(2*tid);
  auto b_base = swizzle(2*tid+1);
  if (2*i < n) sh[a_base] = in[2*i];
  else sh[a_base] = 0;
  if (2*i+1 < n) sh[b_base] = in[2*i+1];
  else sh[b_base] = 0;
  __syncthreads();

  // upsweep
  for (int s = 1; s <= blockSize; s *= 2) {
    auto idx = 2*s*tid + s - 1;
    if (idx+s < 2*blockSize) {
      sh[swizzle(idx+s)] += sh[swizzle(idx)];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_sums[blockIdx.x] = sh[swizzle(2*blockSize-1)];
  }

  // downsweep
  for (int s = blockSize; s > 0; s /= 2) {
    auto idx = 2*s*(tid+1) - 1;
    if (idx + s < 2*blockSize) {
      sh[swizzle(idx+s)] += sh[swizzle(idx)];
    }
    __syncthreads();
  }
  // write results
  if (2*i < n) {
    out[2*i] = sh[a_base];
  }
  if (2*i+1 < n) {
    out[2*i+1] = sh[b_base];
  }
}

auto timeit(std::string const & name, int nrepeat, auto && worker)
{
  using namespace std::chrono;
  std::cout << "Running \'" << name << "\'" << std::endl;
  auto const start_time = high_resolution_clock::now();
  for (int i = 0; i < nrepeat; ++i) {
    if (nrepeat != 1 && nrepeat < 10)
      std::cout << "Repeat " << i + 1 << "/" << nrepeat << std::endl;
    worker();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }
  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );

  auto const end_time = high_resolution_clock::now();
  auto const duration = (duration_cast<microseconds>(end_time - start_time)).count();
  std::cout << "Test \'" << name << "\'"
            << " took " << (double)duration / (double)nrepeat << " [us]";
  if (nrepeat != 1) std::cout << " (average)";
  std::cout << std::endl;
  return duration / (double)nrepeat;
}

struct ApproximateComparator {
  __host__ __device__ bool operator()(thrust::tuple<float, float> const & tup) {
    auto l = thrust::get<0>(tup);
    auto r = thrust::get<1>(tup);
    return (l != r) && (l != 0.f && fabs((l - r)/l) > 1e-6);
  }
};

void compare(std::string name, thrust::device_vector<float> const& y_true,
             thrust::device_vector<float> & y_test,
             auto && worker) {
  thrust::fill(y_test.begin(), y_test.end(), 0);
  worker();
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  // bool not_equal = true;
  auto not_equal = thrust::any_of(thrust::make_zip_iterator(y_true.cbegin(), y_test.cbegin()),
                                  thrust::make_zip_iterator(y_true.cend(), y_test.cend()),
                                  ApproximateComparator{});
  if (not_equal) {
    std::cout << "Test " << name <<" [failed ❌]" << std::endl;
    auto cmp = ApproximateComparator{};
    int n_misses{0};
    for (int i = 0; i < y_true.size() && n_misses < 10; i++) {
      if (cmp(thrust::make_tuple(y_true[i], y_test[i]))) {
        std::cout << "Mismatch at index " << i << ": " << y_true[i] << " != " << y_test[i] << std::endl;
        n_misses++;
      }
    }
    exit(1);
  }
  else {
    std::cout << "Test " << name << " [passed ✅]" << std::endl;
  }
}

void scan(thrust::device_vector<float> & in,
          thrust::device_vector<float> & out,
          int block_size, int tile_size, auto && scan_kernel) {
  auto problem_size = in.size();
  int num_blocks = (problem_size + tile_size - 1) / tile_size;

  thrust::device_vector<float> partial_sums(num_blocks, 0);

  scan_kernel<<<num_blocks, block_size>>>(thrust::raw_pointer_cast(out.data()),
                                          thrust::raw_pointer_cast(in.data()),
                                          problem_size,
                                          thrust::raw_pointer_cast(partial_sums.data()));
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  if (num_blocks > 1) {

    thrust::device_vector<float> partial_sums_of_partial_sums(num_blocks, 0);
    scan(partial_sums, partial_sums_of_partial_sums, block_size, tile_size, scan_kernel);

    add_partial_sums<<<num_blocks, tile_size>>>(thrust::raw_pointer_cast(out.data()),
                                                thrust::raw_pointer_cast(partial_sums_of_partial_sums.data()),
                                                problem_size);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }
}

auto main(int argc, char *argv[]) -> int {

  {
    constexpr int block_size = 256;
    int n = 5*block_size;
    // constexpr int block_size = 32;
    // int n = 2*block_size;
    thrust::device_vector<float> x(n, 1);
    thrust::sequence(x.begin(), x.end(), 1);
    thrust::device_vector<float> y_true(n, 0);
    thrust::device_vector<float> y_test(n, 0);
    thrust::inclusive_scan(x.begin(), x.end(), y_true.begin());

    // void scan(in, out, block_size, tile_size,  scan_kernel) ;
    compare("test naive", y_true, y_test, [&] {
        scan(x, y_test, block_size, block_size, scan_naive<block_size>);
    });
    compare("test more efficient", y_true, y_test, [&] {
        scan(x, y_test, block_size, block_size, scan_more_efficient<block_size>);
    });
    compare("test work efficient", y_true, y_test, [&] {
        scan(x, y_test, block_size, 2*block_size, scan_work_efficient<block_size>);
    });
    compare("test conflict free", y_true, y_test, [&] {
        scan(x, y_test, block_size, 2*block_size, scan_conflict_free<block_size>);
    });
    compare("test conflict free swizzle", y_true, y_test, [&] {
        scan(x, y_test, block_size, 2*block_size, scan_conflict_free_swizzle<block_size>);
    });
    compare("test scan on registers", y_true, y_test, [&] {
        scan(x, y_test, block_size, block_size, scan_on_registers<block_size>);
    });
    compare("scan single pass", y_true, y_test, [&] {
        scan_single_pass(x, y_test, block_size, scan_decoupled_lookback<block_size>);
    });
    compare("scan parallel lookback", y_true, y_test, [&] {
        scan_single_pass(x, y_test, block_size, scan_parallel_lookback<block_size>);
    });
  }

  // benchmarks
  {
    int n = 1 << 24;
    thrust::device_vector<float> x(n);
    thrust::sequence(x.begin(), x.end());
    thrust::device_vector<float> y(n, 0.f);
    constexpr int threads_per_block = 256;
    int n_repeat = 100;
    // int n_repeat = 1;
    int n_blocks = (x.size() + threads_per_block - 1) / threads_per_block;
    std::vector<float> partial_sums(n_blocks, 0);

    thrust::fill(y.begin(), y.end(), 0.f);
    thrust::fill(partial_sums.begin(), partial_sums.end(), 0.f);
    timeit("scan naive", n_repeat, [&] {
        scan_naive<threads_per_block><<<n_blocks, threads_per_block>>>
            (thrust::raw_pointer_cast(y.data()),
             thrust::raw_pointer_cast(x.data()), n,
             thrust::raw_pointer_cast(partial_sums.data()));
      });
    thrust::fill(y.begin(), y.end(), 0.f);
    thrust::fill(partial_sums.begin(), partial_sums.end(), 0.f);
    timeit("scan more efficient", n_repeat, [&] {
        // thrust::fill(y.begin(), y.end(), 0.f);
        scan_more_efficient<threads_per_block><<<n_blocks, threads_per_block>>>
            (thrust::raw_pointer_cast(y.data()),
             thrust::raw_pointer_cast(x.data()), n,
             thrust::raw_pointer_cast(partial_sums.data()));
      });
    thrust::fill(y.begin(), y.end(), 0.f);
    thrust::fill(partial_sums.begin(), partial_sums.end(), 0.f);
    timeit("scan work efficient", n_repeat, [&] {
        scan_work_efficient<threads_per_block>
            <<<n_blocks/2, threads_per_block>>>(thrust::raw_pointer_cast(y.data()),
                                                thrust::raw_pointer_cast(x.data()), n,
                                                thrust::raw_pointer_cast(partial_sums.data()));
      });
    thrust::fill(y.begin(), y.end(), 0.f);
    thrust::fill(partial_sums.begin(), partial_sums.end(), 0.f);
    timeit("scan conflict free", n_repeat, [&] {
        scan_conflict_free<threads_per_block>
            <<<n_blocks/2, threads_per_block>>>(thrust::raw_pointer_cast(y.data()),
                                                thrust::raw_pointer_cast(x.data()), n,
                                                thrust::raw_pointer_cast(partial_sums.data()));
      });
    thrust::fill(y.begin(), y.end(), 0.f);
    thrust::fill(partial_sums.begin(), partial_sums.end(), 0.f);
    timeit("scan conflict free swizzle", n_repeat, [&] {
        scan_conflict_free_swizzle<threads_per_block>
            <<<n_blocks/2, threads_per_block>>>(thrust::raw_pointer_cast(y.data()),
                                                thrust::raw_pointer_cast(x.data()), n,
                                                thrust::raw_pointer_cast(partial_sums.data()));
      });
    thrust::fill(y.begin(), y.end(), 0.f);
    thrust::fill(partial_sums.begin(), partial_sums.end(), 0.f);
    timeit("scan on registers", n_repeat, [&] {
        scan_on_registers<threads_per_block>
            <<<n_blocks, threads_per_block>>>(thrust::raw_pointer_cast(y.data()),
                                              thrust::raw_pointer_cast(x.data()), n,
                                              thrust::raw_pointer_cast(partial_sums.data()));
      });

    std::cout << "--- Full function passes ----" << std::endl;
    // this is not a fair comparison since thrust will do the full scan
    // as opposed to a single step
    timeit("thrust", n_repeat, [&] {
        thrust::inclusive_scan(x.begin(), x.end(), y.begin());
      });
    timeit("scan on registers", n_repeat, [&] {
        scan(x, y, threads_per_block, threads_per_block, scan_on_registers<threads_per_block>);
      });
    timeit("scan decoupled lookback", n_repeat, [&] {
        scan_single_pass(x, y, threads_per_block, scan_decoupled_lookback<threads_per_block>);
      });
    timeit("scan parallel lookback", n_repeat, [&] {
        scan_single_pass(x, y, threads_per_block, scan_parallel_lookback<threads_per_block>);
      });

  }

  return 0;
}
