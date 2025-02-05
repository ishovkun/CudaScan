# Inclusive scan on GPU

This is an experiment in benchmarking inclusive scan kernels on GPU.
I implemented:
- A naive double buffer kernel
- A "more efficient" kernel. This kernel implements a work-efficient scan algorithm.
- A "work efficient" kernel. The algorithm is the same, but each kernel thread processes 
two elements at a time (uses 2*blockSize shared memory). Additionally, it replaces the 
modulo operations with multiplies and achieves less thread divergens.
- A conflict-free kernel. It uses padding to remedy shared memory bank conflicts.
- A conflict-free-swizzle kernel. It uses swizzling instead of padding to fix bank conflicts.

Based on
https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
I also used these slides:
https://people.cs.vt.edu/yongcao/teaching/cs5234/spring2013/slides/Lecture10.pdf


Running 'scan naive'
Test 'scan naive' took 204.83 [us] (average)
Running 'scan more efficient'
Test 'scan more efficient' took 262.94 [us] (average)
Running 'scan work efficient'
Test 'scan work efficient' took 253.41 [us] (average)
Running 'scan conflict free'
Test 'scan conflict free' took 162.62 [us] (average)
Running 'scan conflict free swizzle'
Test 'scan conflict free swizzle' took 162.98 [us] (average)
