# Code Directory 7: Tiled 3D Stencil and Histogram Computation

This directory contains implementations and learnings related to 3D stencil computations and histogram algorithms using shared memory tiling techniques in CUDA.

## Files

### `stencil3d.cu` - Tiled 3D Stencil with Shared Memory
A CUDA implementation of a 3D 7-point stencil computation using shared memory tiling with halo regions. This implementation demonstrates:
- Tiled processing of 3D grids using shared memory
- Proper handling of halo regions for boundary computations
- Fixed version with loops to ensure all elements are processed
- Originally had a bug where not all Z-dimension elements were processed
- Includes both functional and cycle-accurate simulation compatibility

### `8a.cu` - Alternative Stencil Implementation
An alternative or backup implementation of the 3D stencil computation (original version before fixes).

### `histogram.cu` - Simple Histogram Implementation (Primary)
A simplified histogram implementation using direct atomic operations without privatization. This approach successfully works in both functional and cycle-accurate modes and is the recommended implementation.

### `histogram_with_privatization.cu` - Histogram with Shared Memory Privatization
An implementation using privatization with shared memory, where each thread block maintains its own private histogram before aggregating to the global result. This approach works correctly in functional mode but fails in cycle-accurate mode due to synchronization/timing issues in the simulator. This file demonstrates the importance of testing in both functional and cycle-accurate simulation modes, as complex synchronization patterns can behave differently between them.

### `learnings_cuda_stencil.md` - Technical Learnings
A comprehensive document capturing the lessons learned from debugging and optimizing the CUDA stencil code, including:
- Thread block dimension considerations
- Data processing strategies when thread count doesn't match data dimensions
- Shared memory tiling with halo regions
- Functional vs. cycle-accurate simulation modes
- Code changes made and best practices