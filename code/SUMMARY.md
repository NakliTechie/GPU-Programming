# Code Directory Summary

This directory contains implementations and learnings related to CUDA programming concepts, following the book "Programming Massively Parallel Processors".

## Directory Structure

### [1_intro_cuda_concepts](1_intro_cuda_concepts/)
Fundamental concepts of CUDA programming, outlining the interaction between the host (CPU) and the device (GPU).

### [2_vector_addition](2_vector_addition/)
Practical example of vector addition in CUDA, demonstrating the full workflow from data preparation to kernel execution and result verification. Introduces the concept of thread hierarchy for parallel computation.

### [3_matrix_multiplication](3_matrix_multiplication/)
Naive implementation of matrix multiplication, introducing the use of a 2D thread grid to solve 2D problems. Reinforces the concept of mapping a 2D problem to the CUDA execution model.

### [4_execution_model_exploration](4_execution_model_exploration/)
Explores the CUDA execution model in more detail, using `printf` from within a kernel to visualize thread execution and control divergence. Introduces the concepts of warps and latency hiding.

### [5_perf_optimized_matmul](5_perf_optimized_matmul/)
Introduces performance optimization techniques for matrix multiplication. Starts with a tiled approach using shared memory to reduce global memory access, and then further optimizes it by ensuring coalesced memory access patterns.

### [6_convolution](6_convolution/)
Contains implementations related to convolution operations in CUDA, demonstrating how convolution kernels are applied to input data with proper handling of boundary conditions using padding.

### [7_3d_stencil_histogram](7_3d_stencil_histogram/)
Implementations and learnings related to 3D stencil computations and histogram algorithms using shared memory tiling techniques in CUDA.

### [8_reduction_scan](8_reduction_scan/)
Implementations of fundamental parallel algorithms for data aggregation and scan operations in CUDA, demonstrating efficient reduction and prefix sum computations.