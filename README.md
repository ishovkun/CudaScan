# Inclusive scan on GPU

This is an experiment in benchmarking inclusive scan kernels on GPU.
I implemented:
- A double buffer kernel
- A work efficient kernel. This kernel implements a work-efficient scan algorithm but suffers from thread divergence and memory bank conflicts
- A conflict-free kernel. It uses padding to remedy shared memory bank conflicts.
- A conflict-free-swizzle kernel. It uses swizzling instead of padding to fix bank conflicts.
- A scan-on registers kernel. It uses warp synchronization primitives to compute prefix sums and avoid excessive block-thread synchronizations.
- Two decoupled-lookback implementations using previous techniques. The first implementation does a parallel lookback using a block of threads, whereas the latter uses a single warp for the lookback.

Based on
https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
I also used these slides:
https://people.cs.vt.edu/yongcao/teaching/cs5234/spring2013/slides/Lecture10.pdf

# Timings
The experiment is conducted on a A100 GPU.  
| Scan Method           | Number of Passes | Time (us) |
|-----------------------|------------------|-----------|
| Double Buffer         | 3                | 501.477   |
| Single Buffer         | 3                | 582.766   |
| Work Efficient        | 3                | 488.564   |
| Conflict Free Padding | 3                | 448.487   |
| Swizzle               | 3                | 436.744   |
| On Registers          | 3                | 433.118   |
| Block Lookback        | 1                | 380.802   |
| Warp Lookback         | 1                | 277.896   |
| Thrust                | 1                | 269.444   |
