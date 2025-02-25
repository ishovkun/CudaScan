#include <chrono>
#include <iostream>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/iterator/zip_iterator.h>
#include "decoupled_lookback.cuh"
#include "three_pass.cuh"
#include "common.cuh"
#include <iomanip>

auto timeit(std::string const & name, int nrepeat, auto && worker)
{
  using namespace std::chrono;
  // std::cout << "Running \'" << name << "\'" << std::endl;
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
  std::cout << "Benchmark \'" << name << "\'"
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

auto main(int argc, char *argv[]) -> int {

  if (argc == 1)
  {
    // constexpr int block_size = 256;
    constexpr int block_size = 64;
    int n = 5*block_size;
    // constexpr int block_size = 32;
    // int n = 2*block_size;
    thrust::device_vector<float> x(n, 1);
    thrust::sequence(x.begin(), x.end(), 1);
    thrust::device_vector<float> y_true(n, 0);
    thrust::device_vector<float> y_test(n, 0);
    thrust::inclusive_scan(x.begin(), x.end(), y_true.begin());

    // // void scan(in, out, block_size, chunkSize,  scan_kernel) ;
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
    compare("test scan on CUB", y_true, y_test, [&] {
        scan(x, y_test, block_size, block_size, scan_cub<block_size>);
    });
    compare("test fancy scan on CUB", y_true, y_test, [&] {
        constexpr int items_per_thread = 2;
        scan(x, y_test, block_size, items_per_thread*block_size, scan_cub_fancy<block_size, items_per_thread>);
    });
    compare("test fancy scan", y_true, y_test, [&] {
        constexpr int items_per_thread = 2;
        scan(x, y_test, block_size, items_per_thread*block_size, scan_fancy<block_size, items_per_thread>);
    });
    compare("test device CUB", y_true, y_test, [&] {
        cub_device_scan(x, y_test);
    });
    compare("scan single pass", y_true, y_test, [&] {
        scan_single_pass(x, y_test, block_size, scan_serial_lookback<block_size>);
    });
    compare("scan warp lookback", y_true, y_test, [&] {
        scan_single_pass(x, y_test, block_size, scan_warp_lookback<block_size>);
    });
    compare("scan cubbish warp lookback", y_true, y_test, [&] {
        constexpr int itemsPerThread = 2;
        scan_single_pass(x, y_test, block_size, scan_cubbish_warp_lookback<block_size,itemsPerThread>, itemsPerThread);
    });
    compare("scan warp lookback blockload", y_true, y_test, [&] {
        scan_single_pass(x, y_test, block_size, scan_warp_lookback_blockload<block_size>);
    });
    compare("scan block lookback", y_true, y_test, [&] {
        scan_single_pass(x, y_test, block_size, scan_block_lookback<block_size>);
    });
  }

  // benchmarks
  {
    int n = 1 << 24;
    thrust::device_vector<float> x(n);
    thrust::sequence(x.begin(), x.end());
    thrust::device_vector<float> y(n, 0.f);
    constexpr int blockSize = 512;
    int n_repeat = 100;
    if (argc == 2) {
      n_repeat = std::stoi(argv[1]);
    }
    int n_blocks = (x.size() + blockSize - 1) / blockSize;
    std::vector<float> partial_sums(n_blocks, 0);

  //   // thrust::fill(y.begin(), y.end(), 0.f);
  //   // thrust::fill(partial_sums.begin(), partial_sums.end(), 0.f);
  //   // timeit("scan naive", n_repeat, [&] {
  //   //     scan_naive<blockSize><<<n_blocks, blockSize>>>
  //   //         (thrust::raw_pointer_cast(y.data()),
  //   //          thrust::raw_pointer_cast(x.data()), n,
  //   //          thrust::raw_pointer_cast(partial_sums.data()));
  //   //   });
  //   // thrust::fill(y.begin(), y.end(), 0.f);
  //   // thrust::fill(partial_sums.begin(), partial_sums.end(), 0.f);
  //   // timeit("scan more efficient", n_repeat, [&] {
  //   //     // thrust::fill(y.begin(), y.end(), 0.f);
  //   //     scan_more_efficient<blockSize><<<n_blocks, blockSize>>>
  //   //         (thrust::raw_pointer_cast(y.data()),
  //   //          thrust::raw_pointer_cast(x.data()), n,
  //   //          thrust::raw_pointer_cast(partial_sums.data()));
  //   //   });
  //   // thrust::fill(y.begin(), y.end(), 0.f);
  //   // thrust::fill(partial_sums.begin(), partial_sums.end(), 0.f);
  //   // timeit("scan work efficient", n_repeat, [&] {
  //   //     scan_work_efficient<blockSize>
  //   //         <<<n_blocks/2, blockSize>>>(thrust::raw_pointer_cast(y.data()),
  //   //                                             thrust::raw_pointer_cast(x.data()), n,
  //   //                                             thrust::raw_pointer_cast(partial_sums.data()));
  //   //   });
  //   // thrust::fill(y.begin(), y.end(), 0.f);
  //   // thrust::fill(partial_sums.begin(), partial_sums.end(), 0.f);
  //   // timeit("scan conflict free", n_repeat, [&] {
  //   //     scan_conflict_free<blockSize>
  //   //         <<<n_blocks/2, blockSize>>>(thrust::raw_pointer_cast(y.data()),
  //   //                                             thrust::raw_pointer_cast(x.data()), n,
  //   //                                             thrust::raw_pointer_cast(partial_sums.data()));
  //   //   });
  //   // thrust::fill(y.begin(), y.end(), 0.f);
  //   // thrust::fill(partial_sums.begin(), partial_sums.end(), 0.f);
  //   // timeit("scan conflict free swizzle", n_repeat, [&] {
  //   //     scan_conflict_free_swizzle<blockSize>
  //   //         <<<n_blocks/2, blockSize>>>(thrust::raw_pointer_cast(y.data()),
  //   //                                             thrust::raw_pointer_cast(x.data()), n,
  //   //                                             thrust::raw_pointer_cast(partial_sums.data()));
  //   //   });
  //   // thrust::fill(y.begin(), y.end(), 0.f);
  //   // thrust::fill(partial_sums.begin(), partial_sums.end(), 0.f);
  //   // timeit("scan on registers", n_repeat, [&] {
  //   //     scan_on_registers<blockSize>
  //   //         <<<n_blocks, blockSize>>>(thrust::raw_pointer_cast(y.data()),
  //   //                                           thrust::raw_pointer_cast(x.data()), n,
  //   //                                           thrust::raw_pointer_cast(partial_sums.data()));
  //   //   });

  //   std::cout << "--- Full function passes ----" << std::endl;
    // this is not a fair comparison since thrust will do the full scan
    // as opposed to a single step
    timeit("thrust", n_repeat, [&] {
        thrust::inclusive_scan(x.begin(), x.end(), y.begin());
      });
    timeit("scan device CUB", n_repeat, [&] {
        cub_device_scan(x, y);
      });
    // if (blockSize < 512)
    //   timeit("scan 3-pass naive", n_repeat, [&] {
    //       scan(x, y, blockSize, blockSize, scan_naive<blockSize>);
    //     });
    // timeit("scan 3-pass padding", n_repeat, [&] {
    //     scan(x, y, blockSize, 2*blockSize, scan_conflict_free<blockSize>);
    //   });
    timeit("scan 3-pass swizzling", n_repeat, [&] {
        scan(x, y, blockSize, 2*blockSize, scan_conflict_free_swizzle<blockSize>);
      });
    timeit("scan 3-pass on registers", n_repeat, [&] {
        scan(x, y, blockSize, blockSize, scan_on_registers<blockSize>);
      });
    // timeit("scan 3-pass CUB", n_repeat, [&] {
    //     scan(x, y, blockSize, blockSize, scan_cub<blockSize>);
    //   });
    timeit("scan 3-pass CUB fancy", n_repeat, [&] {
        constexpr int itemsPerThread = 4;
        constexpr int blockSize = 256;
        scan(x, y, blockSize, itemsPerThread*blockSize, scan_cub_fancy<blockSize, itemsPerThread>);
      });
    timeit("scan 3-pass fancy", n_repeat, [&] {
        constexpr int itemsPerThread = 4;
        constexpr int blockSize = 256;
        scan(x, y, blockSize, itemsPerThread*blockSize, scan_fancy<blockSize, itemsPerThread>);
      });
    timeit("scan cubbish warp lookback", n_repeat, [&] {
        // this is 149 us
        // constexpr int itemsPerThread = 8;
        // constexpr int blockSize = 128;
        // this is 142 us
        constexpr int itemsPerThread = 16;
        constexpr int blockSize = 64;
        scan_single_pass(x, y, blockSize, scan_cubbish_warp_lookback<blockSize,itemsPerThread>, itemsPerThread);
    });

    // timeit("scan decoupled lookback", n_repeat, [&] {
    //     scan_single_pass(x, y, blockSize, scan_decoupled_lookback<blockSize>);
    //   });
    timeit("scan 1-pass warp lookback", n_repeat, [&] {
        scan_single_pass(x, y, blockSize, scan_warp_lookback<blockSize>);
      });
    // timeit("scan 1-pass block lookback", n_repeat, [&] {
    //     scan_single_pass(x, y, blockSize, scan_block_lookback<blockSize>);
    //   });
    // timeit("scan 1-pass block lookback blockload", n_repeat, [&] {
    //     scan_single_pass(x, y, blockSize, scan_warp_lookback_blockload<blockSize>);
    //   });
    // do not time memory allocations
    // {
    //   constexpr int blockSize = blockSize;
    //   uint32_t* tile_counter;
    //   cudaMalloc(&tile_counter, sizeof(uint32_t));
    //   cudaMemset(tile_counter, 0, sizeof(unsigned int));

    //   int64_t const numBlocks = (x.size() + blockSize - 1) / blockSize;

    //   ScanState *states;
    //   cudaMalloc(&states, numBlocks*sizeof(ScanState));
    //   timeit("scan 1-pass warp lookback noalloc", n_repeat, [&] {
    //       cudaMemset(tile_counter, 0, sizeof(unsigned int));
    //       scan_warp_lookback<blockSize><<<numBlocks, blockSize>>>
    //           (y.data().get(), x.data().get(), x.size(), states, tile_counter);
    //     });

    //   cudaFree(states);
    //   cudaFree(tile_counter);
    // }
  }

  return 0;
}
