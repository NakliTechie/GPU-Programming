// File: vecAdd.cu
// Description: A complete CUDA program to perform vector addition C = A + B.
#include <iostream>
#include <cuda_runtime.h>
#include <cmath> // For fabs

// Helper function to check for CUDA errors
void checkCudaError(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Kernel function that executes on the GPU
// Each thread performs one element-wise addition.
__global__ void vecAdd(float* A, float* B, float* C, int n) {
    // Calculate the global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check to ensure the thread is within the array bounds
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// Main function that runs on the CPU
int main() {
    // --- 1. Define problem size ---
    int n = 1048576; // 2^20 elements
    size_t size = n * sizeof(float);
    std::cout << "Performing vector addition for " << n << " elements." << std::endl;

    // --- 2. Allocate Host (CPU) Memory ---
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    if (h_A == nullptr || h_B == nullptr || h_C == nullptr) {
        std::cerr << "Failed to allocate host memory." << std::endl;
        return -1;
    }

    // --- 3. Initialize Host Data ---
    for (int i = 0; i < n; i++) {
        h_A[i] = sin(i) * sin(i); // Some arbitrary values
        h_B[i] = cos(i) * cos(i);
    }

    // --- 4. Allocate Device (GPU) Memory ---
    float* d_A;
    float* d_B;
    float* d_C;
    checkCudaError(cudaMalloc((void**)&d_A, size), "cudaMalloc d_A");
    checkCudaError(cudaMalloc((void**)&d_B, size), "cudaMalloc d_B");
    checkCudaError(cudaMalloc((void**)&d_C, size), "cudaMalloc d_C");

    // --- 5. Copy Data from Host to Device ---
    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "cudaMemcpy B");

    // --- 6. Define Kernel Launch Parameters ---
    int threadsPerBlock = 256;
    // Integer ceiling division to get the number of blocks needed
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "Launching kernel with " << numBlocks << " blocks of " << threadsPerBlock << " threads." << std::endl;

    // --- 7. Launch the Kernel ---
    vecAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    // --- 8. Copy Results from Device to Host ---
    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "cudaMemcpy C");

    // --- 9. Verify the Results on the CPU ---
    // Since sin^2(x) + cos^2(x) = 1, all elements of C should be 1.0
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_C[i] - 1.0f) > 1e-5) {
            std::cerr << "Verification FAILED at index " << i << ": h_C[" << i << "] = " << h_C[i] << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Verification Successful!" << std::endl;
    }

    // --- 10. Free Memory ---
    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaError(cudaFree(d_A), "cudaFree A");
    checkCudaError(cudaFree(d_B), "cudaFree B");
    checkCudaError(cudaFree(d_C), "cudaFree C");

    return 0;
}