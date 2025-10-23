// Mental Model 3.1

// ===================================
// Host (CPU) Code
// ===================================
main() {
  // 1. Initialize input data on CPU (h_A, h_B)
  // 2. Allocate memory on GPU (d_A, d_B, d_C)
  // 3. Copy inputs to GPU
  
  // 4. Define 2D thread hierarchy and launch kernel
  dim3 threads(16, 16);
  dim3 blocks(N/16, N/16);
  my_2d_kernel<<<blocks, threads>>>(d_A, d_B, d_C);

  // 5. Copy result from GPU back to CPU (h_C_gpu)
  
  // 6. VERIFY:
  //    a. Compute the same result on the CPU (h_C_cpu).
  //    b. Print a sample of h_C_gpu.
  //    c. Print a sample of h_C_cpu.
  //    d. Compare all elements and print SUCCESS/FAILURE.
  
  // 7. Free memory
}

// ===================================
// Device (GPU) Code
// ===================================
__global__ void my_2d_kernel(...) {
  // Calculate 2D thread indices (row, col)
  // Linearize index = row * width + col
  // Perform computation on element at 'index'
}