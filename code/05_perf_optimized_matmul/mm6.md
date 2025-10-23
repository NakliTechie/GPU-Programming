// Mental Model 6.0

// ... Host code ...

// ===================================
// Device (GPU) Code
// ===================================
__global__ void my_optimized_kernel(float* global_A, ...) {
  __shared__ float tile_A[...];
  __shared__ float tile_B[...];

  for (each tile_step) {
    // STRATEGY: Load global data into shared memory in a
    // perfectly COALESCED pattern, even if it means
    // transposing the data on the fly.
    // All 32 threads in a warp should access a contiguous block of global memory.
    tile_A[ty][tx] = global_A[...]; // Coalesced read
    tile_B[ty][tx] = global_B[...]; // Make this coalesced too!

    __syncthreads();

    // Process data from shared memory.
    // This is fast regardless of access pattern (within the tile).
    for (k) { sum += tile_A[...] * tile_B[...]; }

    __syncthreads();
  }

  // ... Write result ...
}