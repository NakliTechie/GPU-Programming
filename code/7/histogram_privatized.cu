// File: histogram_privatized.cu
// Description: A histogram implementation using privatization with shared memory.
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

    // 2. Initialize private histogram to zero in parallel
    // Initialize all bins with only some threads in the block to avoid conflicts
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        private_bins[i] = 0;
    }
    __syncthreads(); // Wait for all threads to finish initialization

    // 3. Each thread processes multiple elements if needed to handle more data than threads
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < N; 
         idx += blockDim.x * gridDim.x) {
        char value = data[idx];
        // Assuming input is lowercase 'a' through 'z'
        if (value >= 'a' && value <= 'z') {
            atomicAdd(&private_bins[value - 'a'], 1);
        }
    }

    // 4. Wait for all threads to finish their private updates
    __syncthreads();

    // 5. Cooperatively add private results to the global bins
    // Use a loop to handle cases where there are more bins than threads per block
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&global_bins[i], private_bins[i]);
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