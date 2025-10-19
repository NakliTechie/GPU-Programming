// Mental Model 2.0

// ===================================
// Host (CPU) Code
// ===================================
main() {
  // 1. Setup, control flow
  host_data_in = ...;
  N = size(host_data_in);

  // 2. Allocate and Transfer to GPU
  cudaMalloc(&device_data_in, N);
  cudaMalloc(&device_data_out, N);
  cudaMemcpy(device_data_in, host_data_in, N, HostToDevice);

  // 3. Define the thread hierarchy and Launch GPU work
  threadsPerBlock = 256;
  numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  my_kernel<<<numBlocks, threadsPerBlock>>>(device_data_in, device_data_out, N);

  // 4. Retrieve results from the GPU
  cudaMemcpy(host_data_out, device_data_out, N, DeviceToHost);

  // 5. Cleanup
  cudaFree(device_data_in);
  cudaFree(device_data_out);
  process_results(host_data_out);
}

// ===================================
// Device (GPU) Code
// ===================================
__global__ void my_kernel(..., int N) {
  // Each thread calculates its unique ID
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread performs its work on its piece of data
  if (idx < N) {
    // ... do computation on device_data[idx] ...
  }
}