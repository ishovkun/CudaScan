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

void compare(std::string name, thrust::device_vector<float> const &y_true,
             thrust::device_vector<float> &y_test, auto &&worker) {
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
    // constexpr int blockSize = 256;
    constexpr int blockSize = 64;
    int n = 5*blockSize;
    // constexpr int blockSize = 32;
    // int n = 2*blockSize;
    thrust::device_vector<float> x(n, 1);
    thrust::sequence(x.begin(), x.end(), 1);
    thrust::device_vector<float> y_true(n, 0);
    thrust::device_vector<float> y_test(n, 0);
    thrust::inclusive_scan(x.begin(), x.end(), y_true.begin());

    // // void scan(in, out, blockSize, chunkSize,  scan_kernel) ;
    compare("test double buffer", y_true, y_test, [&] {
        scan(x, y_test, blockSize, blockSize, scan_double_buffer<blockSize>);
    });
    compare("test single buffer", y_true, y_test, [&] {
        scan(x, y_test, blockSize, blockSize, scan_single_buffer<blockSize>);
    });
    compare("test work efficient", y_true, y_test, [&] {
        constexpr int itemsPerThread = 2;
        scan(x, y_test, blockSize, itemsPerThread*blockSize, scan_work_efficient<blockSize, itemsPerThread>);
    });
    compare("test conflict free padding", y_true, y_test, [&] {
        constexpr int itemsPerThread = 2;
        scan(x, y_test, blockSize, itemsPerThread*blockSize, scan_conflict_free_padding<blockSize, itemsPerThread>);
    });
    compare("test conflict free swizzle", y_true, y_test, [&] {
        constexpr int itemsPerThread = 2;
        scan(x, y_test, blockSize, itemsPerThread*blockSize,
             scan_conflict_free_swizzle<blockSize,itemsPerThread>);
    });
    // compare("test scan on registers", y_true, y_test, [&] {
    //     scan(x, y_test, blockSize, blockSize, scan_on_registers<blockSize>);
    // });
    // compare("test scan on CUB", y_true, y_test, [&] {
    //     scan(x, y_test, blockSize, blockSize, scan_cub<blockSize>);
    // });
    compare("test scan on CUB", y_true, y_test, [&] {
        constexpr int itemsPerThread = 2;
        scan(x, y_test, blockSize, itemsPerThread*blockSize, scan_cub<blockSize, itemsPerThread>);
    });
    compare("test scan on registers", y_true, y_test, [&] {
        constexpr int itemsPerThread = 2;
        scan(x, y_test, blockSize, itemsPerThread*blockSize, scan_on_registers<blockSize, itemsPerThread>);
    });
    compare("test device CUB", y_true, y_test, [&] {
        cub_device_scan(x, y_test);
    });
    compare("scan single pass", y_true, y_test, [&] {
        scan_single_pass(x, y_test, blockSize, scan_serial_lookback<blockSize>);
    });
    compare("scan warp lookback", y_true, y_test, [&] {
        scan_single_pass(x, y_test, blockSize, scan_warp_lookback<blockSize>);
    });
    compare("scan cubbish warp lookback", y_true, y_test, [&] {
        constexpr int itemsPerThread = 2;
        scan_single_pass(x, y_test, blockSize, scan_cubbish_warp_lookback<blockSize,itemsPerThread>, itemsPerThread);
    });
    compare("scan warp lookback blockload", y_true, y_test, [&] {
        scan_single_pass(x, y_test, blockSize, scan_warp_lookback_blockload<blockSize>);
    });
    compare("scan block lookback", y_true, y_test, [&] {
        constexpr int itemsPerThread = 2;
        scan_single_pass(x, y_test, blockSize, scan_block_lookback<blockSize, itemsPerThread>);
    });
    compare("scan fancy lookback", y_true, y_test, [&] {
        constexpr int itemsPerThread = 2;
        scan_single_pass(x, y_test, blockSize, fancy_scan_lookback<blockSize,itemsPerThread>, itemsPerThread);
    });
  }

  // benchmarks
  {
    int n = 1 << 24;
    thrust::device_vector<float> x(n);
    thrust::sequence(x.begin(), x.end());
    thrust::device_vector<float> y(n, 0.f);
    // constexpr int blockSize = 128;
    int n_reps = 200;
    if (argc == 2) {
      n_reps = std::stoi(argv[1]);
    }
    // i wanna vary block size, so let's just make
    // partial sums of the size of the input
    std::vector<float> partial_sums(n, 0);

    // thrust::fill(y.begin(), y.end(), 0.f);
    // thrust::fill(partial_sums.begin(), partial_sums.end(), 0.f);

  //   std::cout << "--- Full function passes ----" << std::endl;
    // this is not a fair comparison since thrust will do the full scan
    // as opposed to a single step
    timeit("thrust", n_reps, [&] {
        thrust::inclusive_scan(x.begin(), x.end(), y.begin());
      });
    // timeit("scan device CUB", n_reps, [&] {
    //     cub_device_scan(x, y);
    //   });
    timeit("scan 3-pass double buffer", n_reps, [&] {
        constexpr int blockSize = 256;
        scan(x, y, blockSize, blockSize, scan_double_buffer<blockSize>);
      });
    timeit("scan 3-pass single buffer", n_reps, [&] {
        constexpr int blockSize = 256;
        scan(x, y, blockSize, blockSize, scan_single_buffer<blockSize>);
      });
    timeit("scan 3-pass work efficient", n_reps, [&] {
        constexpr int blockSize = 256;
        constexpr int itemsPerThread = 4;
        scan(x, y, blockSize, itemsPerThread*blockSize,
             scan_work_efficient<blockSize, itemsPerThread>);
      });
    timeit("scan 3-pass conflict free padding", n_reps, [&] {
        constexpr int blockSize = 256;
        constexpr int itemsPerThread = 4;
        scan(x, y, blockSize, itemsPerThread*blockSize,
             scan_conflict_free_padding<blockSize, itemsPerThread>);
      });
    timeit("scan 3-pass swizzle", n_reps, [&] {
        constexpr int blockSize = 256;
        constexpr int itemsPerThread = 4;
        scan(x, y, blockSize, itemsPerThread*blockSize,
             scan_conflict_free_swizzle<blockSize, itemsPerThread>);
      });
    // timeit("scan 3-pass on registers", n_reps, [&] {
    //     scan(x, y, blockSize, blockSize, scan_on_registers<blockSize>);
    //   });
    // timeit("scan 3-pass CUB", n_reps, [&] {
    //     scan(x, y, blockSize, blockSize, scan_cub<blockSize>);
    //   });
    timeit("scan 3-pass on registers", n_reps, [&] {
        constexpr int itemsPerThread = 4;
        constexpr int blockSize = 256;
        scan(x, y, blockSize, itemsPerThread*blockSize, scan_on_registers<blockSize, itemsPerThread>);
      });
    timeit("scan 3-pass CUB", n_reps, [&] {
        constexpr int itemsPerThread = 4;
        constexpr int blockSize = 256;
        scan(x, y, blockSize, itemsPerThread*blockSize, scan_cub<blockSize, itemsPerThread>);
      });

    // ==================== Single pass ====================
    // timeit("scan cubbish warp lookback", n_reps, [&] {
    //     // this is 149 us
    //     // constexpr int itemsPerThread = 8;
    //     // constexpr int blockSize = 128;
    //     // this is 142 us
    //     constexpr int itemsPerThread = 16;
    //     constexpr int blockSize = 64;
    //     scan_single_pass(x, y, blockSize, scan_cubbish_warp_lookback<blockSize,itemsPerThread>, itemsPerThread);
    // });

    // timeit("scan decoupled lookback", n_reps, [&] {
    //     scan_single_pass(x, y, blockSize, scan_decoupled_lookback<blockSize>);
    //   });
    // timeit("scan 1-pass warp lookback", n_reps, [&] {
    //     scan_single_pass(x, y, blockSize, scan_warp_lookback<blockSize>);
    //   });
    timeit("scan 1-pass block lookback", n_reps, [&] {
        constexpr int itemsPerThread = 16;
        constexpr int blockSize = 128;
        scan_single_pass(x, y, blockSize, scan_block_lookback<blockSize,itemsPerThread>, itemsPerThread);
      });
    timeit("scan 1-pass warp lookback", n_reps, [&] {
        constexpr int itemsPerThread = 16;
        constexpr int blockSize = 128;
        scan_single_pass(x, y, blockSize, fancy_scan_lookback<blockSize,itemsPerThread>, itemsPerThread);
      });
    // timeit("scan 1-pass block lookback blockload", n_reps, [&] {
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
    //   timeit("scan 1-pass warp lookback noalloc", n_reps, [&] {
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
