// Mental Model 8.0

// ===================================
// Device (GPU) Code
// ===================================
__global__ void my_advanced_kernel(...) {
  // STRATEGY 1: Pure Shared Memory Tiling
  // Pro: Conceptually simple extension of 2D tiling.
  // Con: Uses a lot of shared memory (volume grows cubically).
  __shared__ float tile_3D[...][...][...];
  // ... load 3D tile, sync, compute, sync ...

  // --- OR ---

  // STRATEGY 2: Thread Coarsening + Register/Shared Tiling
  // Pro: Greatly reduces shared memory pressure.
  // Con: More complex kernel logic.
  __shared__ float tile_2D[...][...];
  float z_column_registers[...];

  for (z_slice in Z_dimension) {
      // Load 2D slice into shared memory
      tile_2D[...][...] = ...;
      __syncthreads();

      // Compute using 2D shared tile and private Z-registers
      // ...
      
      __syncthreads();
  }
}