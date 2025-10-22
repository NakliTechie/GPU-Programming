# Code Directory Summary

This directory contains implementations and learnings related to CUDA programming concepts, following the book "Programming Massively Parallel Processors".

## Directory Structure

### [intro_cuda_concepts](intro_cuda_concepts/)
Fundamental concepts of CUDA programming, outlining the interaction between the host (CPU) and the device (GPU).

### [vector_addition](vector_addition/)
Practical example of vector addition in CUDA, demonstrating the full workflow from data preparation to kernel execution and result verification. Introduces the concept of thread hierarchy for parallel computation.

### [matrix_multiplication](matrix_multiplication/)
Naive implementation of matrix multiplication, introducing the use of a 2D thread grid to solve 2D problems. Reinforces the concept of mapping a 2D problem to the CUDA execution model.

### [execution_model_exploration](execution_model_exploration/)
Explores the CUDA execution model in more detail, using `printf` from within a kernel to visualize thread execution and control divergence. Introduces the concepts of warps and latency hiding.

### [perf_optimized_matmul](perf_optimized_matmul/)
Introduces performance optimization techniques for matrix multiplication. Starts with a tiled approach using shared memory to reduce global memory access, and then further optimizes it by ensuring coalesced memory access patterns.

### [convolution](convolution/)
Contains implementations related to convolution operations in CUDA, demonstrating how convolution kernels are applied to input data with proper handling of boundary conditions using padding.

### [3d_stencil_histogram](3d_stencil_histogram/)
Implementations and learnings related to 3D stencil computations and histogram algorithms using shared memory tiling techniques in CUDA.

### [reduction_scan](reduction_scan/)
Implementations of fundamental parallel algorithms for data aggregation and scan operations in CUDA, demonstrating efficient reduction and prefix sum computations.