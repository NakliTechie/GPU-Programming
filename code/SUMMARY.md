# Code Directory Summary

This directory contains implementations and learnings related to CUDA programming concepts, following the book "Programming Massively Parallel Processors".

## Directory Structure

### [01_intro_cuda_concepts](01_intro_cuda_concepts/)
Fundamental concepts of CUDA programming, outlining the interaction between the host (CPU) and the device (GPU).

### [02_vector_addition](02_vector_addition/)
Practical example of vector addition in CUDA, demonstrating the full workflow from data preparation to kernel execution and result verification. Introduces the concept of thread hierarchy for parallel computation.

### [03_matrix_multiplication](03_matrix_multiplication/)
Naive implementation of matrix multiplication, introducing the use of a 2D thread grid to solve 2D problems. Reinforces the concept of mapping a 2D problem to the CUDA execution model.

### [04_execution_model_exploration](04_execution_model_exploration/)
Explores the CUDA execution model in more detail, using `printf` from within a kernel to visualize thread execution and control divergence. Introduces the concepts of warps and latency hiding.

### [05_perf_optimized_matmul](05_perf_optimized_matmul/)
Introduces performance optimization techniques for matrix multiplication. Starts with a tiled approach using shared memory to reduce global memory access, and then further optimizes it by ensuring coalesced memory access patterns.

### [06_convolution](06_convolution/)
Contains implementations related to convolution operations in CUDA, demonstrating how convolution kernels are applied to input data with proper handling of boundary conditions using padding.

### [07_3d_stencil_histogram](07_3d_stencil_histogram/)
Implementations and learnings related to 3D stencil computations and histogram algorithms using shared memory tiling techniques in CUDA.

### [08_reduction_scan](08_reduction_scan/)
Implementations of fundamental parallel algorithms for data aggregation and scan operations in CUDA, demonstrating efficient reduction and prefix sum computations.

### [09_parallel_sorting_algorithms](09_parallel_sorting_algorithms/)
Implementations of parallel sorting algorithms in CUDA, demonstrating efficient parallel sorting techniques like merge sort and radix sort.