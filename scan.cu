#include <iostream>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    std::cout << "GPU assert failed" << std::endl;
    if (abort) exit(code);
  }
}

template <int blockSize>
__global__ void scan(float* out, float* in, int n) {
  auto tid = threadIdx.x;
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sh[2*blockSize];
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
}

template <int blockSize>
__global__ void scan_work_efficient(float* out, float const* in, int n) {
  auto tid = threadIdx.x;
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sh[blockSize];
  if (i < n) sh[tid] = in[i];
  else       sh[tid] = 0.f;
  __syncthreads();

  // upsweep
  /*
    | idx  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 1 | 13 | 14 | 15 |
    | init | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |  1 |  1 | 1 |  1 |  1 |  1 |
    | s=1  | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 |  1 |  2 | 1 |  2 |  1 |  2 |
    | s=2  | 1 | 2 | 1 | 4 | 1 | 2 | 1 | 4 | 1 | 2 |  1 |  4 | 1 |  2 |  1 |  4 |
    | s=4  | 1 | 2 | 1 | 4 | 1 | 2 | 1 | 8 | 1 | 2 |  1 |  4 | 1 |  2 |  1 |  8 |
    | s=8  | 1 | 2 | 1 | 4 | 1 | 2 | 1 | 8 | 1 | 2 |  1 |  4 | 1 |  2 |  1 | 16 |
  */
  for (int s = 1; s < blockSize; s *= 2) {
    if ((tid+1) % (2*s) == 0) {
      sh[tid] += sh[tid-s];
    }
    __syncthreads();
  }
  __syncthreads();
  // downsweep
  /*
downsweep
  | idx  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
  | init | 1 | 2 | 1 | 4 | 1 | 2 | 1 | 8 | 1 |  2 |  1 |  4 |  1 |  2 |  1 | 16 |
         | |------------------------------>>>>>>>>---------------------------\
  | le   | 1 | 2 | 1 | 4 | 1 | 2 | 1 | 8 | 1 |  2 |  1 |  4 |  1 |  2 |  1 |  1 |
         |                            |--------------------------------------\
  | s=8  | 1 | 2 | 1 | 4 | 1 | 2 | 1 | 1 | 1 |  2 |  1 |  4 |  1 |  2 |  1 |  8 |
         |            |--------------\                   |------------------\
  | s=4  | 1 | 2 | 1 | 1 | 1 | 2 | 1 | 5 | 1 |  2 |  1 |  8 |  1 |  2 |  1 | 12 |
         |    |------\        |------\         |--------\          |---------\
  | s=2  | 1 | 1 | 1 | 3 | 1 | 5 | 1 | 7 | 1 |  8 |  1 | 10 |  1 | 12 |  1 | 14 |
         ||---\   |---\   |---\   |---\   |----\    |---\     |----\    |----\
  | s=1  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 10 | 11 | 12 | 13 | 14 | 15 |
   */
  if (tid == blockSize - 1) sh[tid] = sh[0]; // last element (le)
  for (int s = blockSize/2; s > 0; s /= 2) {
    if ((tid+1) % (2*s) == 0) {
      auto tmp = sh[tid];
      sh[tid] += sh[tid-s];
      sh[tid-s] = tmp;
    }
    __syncthreads();
  }
  if (i < n)
    out[i] = sh[tid];
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
  }

  auto const end_time = high_resolution_clock::now();
  auto const duration = (duration_cast<microseconds>(end_time - start_time)).count();
  std::cout << "Test \'" << name << "\'"
            << " took " << (double)duration / (double)nrepeat << " [us]";
  if (nrepeat != 1) std::cout << " (average)";
  std::cout << std::endl;
  return duration / (double)nrepeat;
}

void compare(std::string name, thrust::device_vector<float> const& y_true,
             thrust::device_vector<float> & y_test,
             auto && worker) {
  thrust::fill(y_test.begin(), y_test.end(), 0);
  worker();
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  if (y_true != y_test) {
    std::cout << "Test " << name <<" failed" << std::endl;
    exit(1);
  }
  else {
    std::cout << "Test " << name << " passed" << std::endl;

  }
}

auto main(int argc, char *argv[]) -> int {

  int n = 32;
  thrust::device_vector<float> x(32, 1);
  thrust::device_vector<float> y_true(32, 0);
  thrust::device_vector<float> y_test(32, 0);
  constexpr int block_size = 32;
  int n_blocks = (n + block_size - 1) / block_size;
  scan<block_size><<<n_blocks, block_size>>>(
      thrust::raw_pointer_cast(y_true.data()),
      thrust::raw_pointer_cast(x.data()), n);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  compare("test work efficient", y_true, y_test, [&] {
      scan_work_efficient<block_size><<<n_blocks, block_size>>>(
          thrust::raw_pointer_cast(y_test.data()),
          thrust::raw_pointer_cast(x.data()), x.size());
    });
  // thrust::fill(y.begin(), y.end(), 0);
  // scan_work_efficient<block_size><<<n_blocks, block_size>>>(
  //     thrust::raw_pointer_cast(y_test.data()),
  //     thrust::raw_pointer_cast(x.data()), n);
  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );

  // for (int i = 0; i < n; i++)
  //   std::cout << i << "\t";
  // std::cout << std::endl;
  // for (auto val : x)
  //   std::cout << val << "\t";
  // std::cout << std::endl;
  // for (auto val : y)
  //   std::cout << val << "\t";
  // std::cout << std::endl;


  return 0;
}
