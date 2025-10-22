# CUDA Learning Document: Stencil Computation and Shared Memory Optimization

## Overview
This document captures the key learnings from debugging and optimizing a 3D tiled stencil computation using shared memory with halos in CUDA.

## Original Problem Statement
We had a CUDA program (stencil3d.cu) that implemented a 3D 7-point stencil computation using shared memory tiling. The program had a verification mismatch between CPU and GPU results, with the center voxel values matching but verification failing when comparing all grid points.

## Key Learnings

### 1. Thread Block Dimensions vs. Data Dimensions
- When your thread block dimensions don't match your data dimensions, some data may not be processed
- In our case: thread block Z-dimension was `BLOCK_DIM / 8` = 2, but shared memory tile Z-dimension was `BLOCK_DIM` = 18
- This meant only 2 of 18 Z-dimension elements were being processed per tile

### 2. Using Loops Within Threads for Processing Multiple Elements
- When you have fewer threads than elements to process in a dimension, use loops within each thread
- Pattern: `for (int i = tid; i < num_elements; i += blockDim.x)` to distribute work across threads
- This ensures all elements are processed even when there aren't enough threads

### 3. Importance of Halo Regions in Tiled Stencil Computations
- Halo regions (border areas) contain data from neighboring tiles needed for stencil computations
- Critical for correctness at tile boundaries
- Each tile must include halo data for neighbors that will be needed in computations

### 4. Functional vs. Cycle-Accurate Simulation Modes
- **Functional Mode**: Fast, good for correctness verification
- **Cycle-Accurate Mode**: Detailed hardware simulation, slower but provides performance insights
- Complex kernels with more loops can run much slower in cycle-accurate mode

### 5. Performance Considerations
- Reducing problem size (N=64 to N=32) can make cycle-accurate simulation more practical
- Algorithmic correctness should be verified in functional mode first
- More complex kernels with loops can significantly impact simulation time in cycle-accurate mode

### 6. Debugging Strategy
- Start with functional mode for correctness verification
- Use cycle-accurate mode for performance analysis once correctness is established
- When verification fails, check if all data elements are being processed in the kernel
- Verify thread block dimensions match work distribution needs

### 7. Stencil vs. Convolution Key Differences
- **Stencil**: Used for scientific simulations, typically with uniform operations
- **Convolution**: Used for feature detection, with specific filter weights
- Both involve weighted sums of neighbors but serve different purposes

## Common CUDA Debugging Techniques Applied
1. Examined thread indexing and data mapping
2. Checked boundary conditions and memory access patterns
3. Verified work distribution across threads
4. Used both simulation modes for validation

## Best Practices Learned
1. Ensure your thread block configuration matches your data processing needs
2. When using shared memory tiles, verify all elements in the tile get processed
3. Use loops within threads when there are more elements than threads in any dimension
4. Always verify correctness in functional mode before performance analysis
5. Consider problem size when using cycle-accurate simulators

## Code Changes Made

### Original Code (Incomplete Processing)
```cuda
// Loading data into shared memory - processed only one element per thread
int ix = blockIdx.x * TILE_DIM + tx - RADIUS;
int iy = blockIdx.y * TILE_DIM + ty - RADIUS;
int iz = blockIdx.z * TILE_DIM + tz - RADIUS;

if (ix >= 0 && ix < N && iy >= 0 && iy < N && iz >= 0 && iz < N) {
    tile[tz][ty][tx] = in[iz * N * N + iy * N + ix];
} else {
    tile[tz][ty][tx] = 0.0f;
}

// Processing output - computed only one output per thread
if (tx >= RADIUS && tx < BLOCK_DIM - RADIUS &&
    ty >= RADIUS && ty < BLOCK_DIM - RADIUS &&
    tz >= RADIUS && tz < BLOCK_DIM - RADIUS)
{
    int out_x = blockIdx.x * TILE_DIM + (tx - RADIUS);
    int out_y = blockIdx.y * TILE_DIM + (ty - RADIUS);
    int out_z = blockIdx.z * TILE_DIM + (tz - RADIUS);
    
    // ... computation code ...
}
```

### Fixed Code (Complete Processing)
```cuda
// Loading data into shared memory - each thread processes multiple elements
for (int z_iter = tz; z_iter < BLOCK_DIM; z_iter += blockDim.z) {
    int iz = blockIdx.z * TILE_DIM + z_iter - RADIUS;
    int ix = blockIdx.x * TILE_DIM + tx - RADIUS;
    int iy = blockIdx.y * TILE_DIM + ty - RADIUS;

    if (ix >= 0 && ix < N && iy >= 0 && iy < N && iz >= 0 && iz < N) {
        tile[z_iter][ty][tx] = in[iz * N * N + iy * N + ix];
    } else {
        tile[z_iter][ty][tx] = 0.0f;
    }
}

// Processing output - each thread processes multiple outputs
for (int z_iter = tz + RADIUS; z_iter < BLOCK_DIM - RADIUS; z_iter += blockDim.z) {
    if (tx >= RADIUS && tx < BLOCK_DIM - RADIUS &&
        ty >= RADIUS && ty < BLOCK_DIM - RADIUS)
    {
        int out_x = blockIdx.x * TILE_DIM + (tx - RADIUS);
        int out_y = blockIdx.y * TILE_DIM + (ty - RADIUS);
        int out_z = blockIdx.z * TILE_DIM + (z_iter - RADIUS);
        
        // ... computation code ...
    }
}
```

### Key Change Pattern
The essential pattern added was: `for (int i = tid; i < num_elements; i += blockDim.x)` to ensure all elements are processed even when there are more elements than threads in any dimension.


## Additional Learnings from Histogram Investigation

### 1. Functional vs. Cycle-Accurate Simulation Differences for Complex Algorithms
We also investigated histogram implementations and discovered that some algorithms that work perfectly in functional mode can fail in cycle-accurate mode due to:
- Different thread scheduling behavior
- Precise timing of operations that differs between simulation modes
- Synchronization points that behave differently under detailed timing models

### 2. Different Histogram Approaches
We evaluated multiple approaches for histogram computation:
- Privatization approach: Uses shared memory for thread-block local histograms before aggregating to global results
  - Pros: Potentially more efficient by reducing global memory atomic contention
  - Cons: Complex synchronization vulnerable to timing issues in detailed simulators
  
- Direct approach (implemented in histogram.cu): Each thread atomically updates the global histogram directly
  - Pros: Simpler, more robust across different simulation modes
  - Cons: Higher global memory contention

### 3. Robust Implementation Selection
When multiple implementations exist, the primary filename should be given to the implementation that works correctly in both simulation modes, ensuring reliability for users who need consistent behavior across different simulation environments.

### 4. Simulation Mode Testing
Always test complex CUDA algorithms in both functional and cycle-accurate simulation modes when possible, as subtle race conditions or synchronization issues can manifest only in one mode.

### 5. Simulation Mode Compatibility
Some algorithms that work perfectly in functional mode can fail in cycle-accurate mode due to different thread scheduling behavior, precise timing differences, and synchronization points that behave differently under detailed timing models.

### 6. Understanding Atomic Operations and Privatization
- **Atomic Operations**: Ensure memory updates happen indivisibly, preventing race conditions when multiple threads access the same memory location. Example: `atomicAdd()` performs a read-modify-write operation that can't be interrupted by other threads, which is critical for correctness when multiple threads need to update shared data structures like histograms.
  
- **Privatization**: A performance optimization technique where each thread block maintains its own local data structures (like histograms in shared memory) before combining results. This reduces contention on global memory by having threads update faster shared memory first. However, it requires careful synchronization to coordinate local computation with global aggregation.

### 7. Algorithm Selection for Learning Projects
When algorithms demonstrate important concepts but fail in certain simulation modes, it can be valuable to keep both implementations in learning projects to showcase the differences and teach about the importance of testing in different environments.

### 8. Simulation vs. Real Hardware Behavior
The privatization approach with shared memory histograms was conceptually correct and more efficient than the direct approach:
- Worked correctly on actual hardware (tested on GTX 1650)
- Worked correctly in functional simulation mode
- Failed in cycle-accurate simulation mode due to timing-sensitive synchronization differences

This highlighted that while atomic operations and privatization are valid CUDA programming techniques, the precise timing requirements of cycle-accurate simulators can expose dependencies that don't manifest on real hardware, emphasizing the importance of testing across different environments.