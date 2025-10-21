// File: histogram_privatized_minimal_atomics.cu
// Description: A histogram implementation using privatization with minimal atomic usage.
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

#define NUM_BINS 26 // For letters 'a' through 'z'

void checkCudaError(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void histogram_kernel(const char* data, int* global_bins, int N) {
    // 1. Declare private histogram in shared memory
    __shared__ int private_bins[NUM_BINS];
    __shared__ bool is_initialized;

    // 2. Initialize private histogram to zero - only one thread does this to prevent race conditions
    if (threadIdx.x == 0) {
        is_initialized = false;
    }
    __syncthreads(); // Wait for the flag to be set

    // Only initialize once all threads have reached this point
    if (threadIdx.x < NUM_BINS) {
        private_bins[threadIdx.x] = 0;
    }
    __syncthreads(); // Wait for all bins to be zeroed
    
    if (threadIdx.x == 0) {
        is_initialized = true;
    }
    __syncthreads(); // Wait for initialization to be marked as complete

    // 3. Each thread processes its data element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && is_initialized) {  // Only process after initialization is confirmed
        char value = data[idx];
        // Assuming input is lowercase 'a' through 'z'
        if (value >= 'a' && value <= 'z') {
            // Use a simple increment instead of atomic, since only one thread accesses each value
            // But still need atomic for the shared memory access to be safe
            atomicAdd(&private_bins[value - 'a'], 1);
        }
    }

    // 4. Wait for all threads to finish their private updates
    __syncthreads();

    // 5. Cooperatively add private results to the global bins
    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&global_bins[threadIdx.x], private_bins[threadIdx.x]);
    }
}

int main() {
    int N = 8192; // Smaller problem size for better compatibility
    size_t data_size = N * sizeof(char);
    size_t bins_size = NUM_BINS * sizeof(int);

    std::vector<char> h_data(N);
    std::vector<int> h_bins_gpu(NUM_BINS, 0);
    std::vector<int> h_bins_cpu(NUM_BINS, 0);

    // Initialize data with a predictable pattern: 'a', 'b', 'c', ...
    for (int i = 0; i < N; ++i) {
        h_data[i] = 'a' + (i % NUM_BINS);
    }

    char* d_data;
    int* d_bins;
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_bins, bins_size);

    cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice);
    // Important: Initialize global bins to zero before kernel launch!
    cudaMemset(d_bins, 0, bins_size);

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    // Limit blocks to reduce resource usage
    if (numBlocks > 64) numBlocks = 64;

    std::cout << "Launching privatized histogram kernel with N=" << N << ", " << numBlocks << " blocks..." << std::endl;
    histogram_kernel<<<numBlocks, threadsPerBlock>>>(d_data, d_bins, N);
    cudaDeviceSynchronize();
    std::cout << "Kernel finished." << std::endl;

    cudaMemcpy(h_bins_gpu.data(), d_bins, bins_size, cudaMemcpyDeviceToHost);

    // Verification on CPU
    for (int i = 0; i < N; ++i) {
        char value = h_data[i];
        if (value >= 'a' && value <= 'z') {
            h_bins_cpu[value - 'a']++;
        }
    }

    std::cout << "\n--- Sample Results (First 4 bins) ---" << std::endl;
    std::cout << "Bin | GPU Result | CPU Result" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << (char)('a' + i) << "   | " << h_bins_gpu[i] << "\t | " << h_bins_cpu[i] << std::endl;
    }

    bool success = true;
    for (int i = 0; i < NUM_BINS; ++i) {
        if (h_bins_gpu[i] != h_bins_cpu[i]) {
            success = false;
            break;
        }
    }
    std::cout << (success ? "\nVerification Successful!" : "\nVerification FAILED!") << std::endl;

    cudaFree(d_data);
    cudaFree(d_bins);

    return 0;
}