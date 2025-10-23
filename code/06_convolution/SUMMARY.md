# Code Directory 6: Convolution Implementation

This directory contains implementations related to convolution operations in CUDA.

## Files

### `convolution.cu` - CUDA Convolution Implementation
A CUDA implementation demonstrating convolution operations, which apply a weighted filter by computing dot products between the kernel and input data. This implementation shows:
- How convolution kernels are applied to input data
- Proper handling of boundary conditions using padding
- Efficient memory access patterns for convolution operations
- Comparison between theoretical convolution and practical implementation

### `mm7.md` - Mental Model 7
A documentation file explaining the mental model for convolution operations in CUDA, including:
- Host (CPU) code structure for setting up convolution
- Device (GPU) code structure for parallel computation
- Use of constant memory for filter storage
- Shared memory tiling with halo regions for efficient computation
- Thread block configuration for input tile processing