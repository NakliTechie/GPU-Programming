// Mental Model 7.0

// ===================================
// Host (CPU) Code
// ===================================
main() {
  // ...
  // Copy filter data to __constant__ memory
  cudaMemcpyToSymbol(device_filter, host_filter, ...);

  // Launch kernel with blocks sized for INPUT tile
  dim3 blockDim(TILE_WIDTH + 2*R, TILE_WIDTH + 2*R);
  dim3 gridDim(OUTPUT_W / TILE_WIDTH, OUTPUT_H / TILE_WIDTH);
  my_conv_kernel<<<gridDim, blockDim>>>(...);
  // ...
}

// ===================================
// Device (GPU) Code
// ===================================
__constant__ float device_filter[...];

__global__ void my_conv_kernel(...) {
  // 1. Define shared memory tile sized to include HALO
  __shared__ float tile[TILE_WIDTH + 2*R][TILE_WIDTH + 2*R];

  // 2. Each thread loads one element of the larger input tile (including halo)
  //    into shared memory.
  tile[ty][tx] = global_input[...];
  
  __syncthreads();

  // 3. ONLY the "inner" threads that correspond to an output element do work.
  if (thread corresponds to output pixel) {
    // 4. Compute by looping over the filter and reading from the shared memory tile.
    //    Access filter from __constant__ memory.
    for (i, j) { sum += tile[...]*device_filter[...]; }

    // 5. Write result to global memory.
    global_output[...] = sum;
  }
}