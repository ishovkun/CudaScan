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
