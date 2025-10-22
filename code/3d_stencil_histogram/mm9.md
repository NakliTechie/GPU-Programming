// Mental Model 9.0

// ... Host code ...

// ===================================
// Device (GPU) Code
// ===================================
__global__ void my_privatized_kernel(..., int* global_results) {
  // 1. Define a private, per-block data structure in shared memory.
  __shared__ int private_results[...];

  // 2. Initialize the private structure in parallel.
  private_results[threadIdx.x] = 0;
  __syncthreads(); // Barrier 1: Ensure initialization is complete.

  // 3. Each thread performs work and updates the PRIVATE structure.
  // Atomics here are fast because they are on shared memory.
  int idx = ...;
  if (idx < N) {
      ...
      atomicAdd(&private_results[...], 1);
  }
  
  // 4. BARRIER 2: Ensure all private updates are complete.
  __syncthreads();

  // 5. Cooperatively commit the private results to the GLOBAL structure.
  // Atomics here are slow, but contention is massively reduced.
  atomicAdd(&global_results[threadIdx.x], private_results[threadIdx.x]);
}