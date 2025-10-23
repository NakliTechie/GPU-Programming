// Mental Model 5.0

// ... Host code ...
my_tiled_kernel<<<gridDim, blockDim>>>(...);

// ===================================
// Device (GPU) Code
// ===================================
__global__ void my_tiled_kernel(float* global_A, ...) {
  // 1. Define tiles in fast, on-chip shared memory
  __shared__ float tile_A[...];
  __shared__ float tile_B[...];

  // 2. Loop over large global matrices in tile-sized steps
  for (each tile_step) {
    // 3. Cooperatively load one tile from slow global memory
    //    into fast shared memory.
    tile_A[...] = global_A[...];
    tile_B[...] = global_B[...];

    // 4. BARRIER: Wait for ALL threads in the block to finish loading.
    __syncthreads();

    // 5. All threads process the loaded tiles using FAST reads from shared memory.
    for (each element_in_tile) {
      sum += tile_A[...] * tile_B[...];
    }

    // 6. BARRIER: Wait for ALL threads to finish using the tiles.
    __syncthreads();
  }

  // 7. Write final result from registers to slow global memory.
  global_C[...] = sum;
}