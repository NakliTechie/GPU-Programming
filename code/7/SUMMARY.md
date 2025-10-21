# Code Directory 7: Tiled 3D Stencil and Histogram Computation

This directory contains implementations and learnings related to 3D stencil computations and histogram algorithms using shared memory tiling techniques in CUDA.

## Files

### `stencil3d.cu` - Tiled 3D Stencil with Shared Memory (Fixed Version)
A CUDA implementation of a 3D 7-point stencil computation using shared memory tiling with halo regions. This implementation demonstrates:
- Tiled processing of 3D grids using shared memory
- Proper handling of halo regions for boundary computations
- Fixed version with loops to ensure all elements are processed
- Originally had a bug where not all Z-dimension elements were processed
- Includes both functional and cycle-accurate simulation compatibility

### `8a.cu` - 3D Stencil with Original Bug
An implementation of the 3D stencil computation with the original threading bug where not all data elements get processed. This produces incorrect results and demonstrates the importance of proper thread-to-data mapping in CUDA kernels.

### `histogram.cu` - Simple Histogram Implementation (Primary)
A simplified histogram implementation using direct atomic operations without privatization. Each thread processes one element and directly updates the global histogram using atomic operations. This approach successfully works in both functional and cycle-accurate modes and on real hardware. It is the recommended implementation due to its simplicity and robustness across different execution environments.

### `histogram_with_privatization.cu` - Histogram with Shared Memory Privatization
An implementation using privatization where each thread block maintains its own private histogram in shared memory before aggregating to the global result. This approach reduces contention on global memory by having threads update the faster shared memory first, then combining results. This implementation:
- Works correctly on real hardware (tested on Windows machine with GTX 1650)
- Works correctly in functional simulation mode
- Fails in cycle-accurate simulation mode due to synchronization/timing issues in the simulator
This demonstrates the importance of testing in both functional and cycle-accurate simulation modes, as complex synchronization patterns can behave differently between them. The privatization approach is conceptually better for performance but more sensitive to timing differences in simulators.

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