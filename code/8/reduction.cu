// File: reduction.cu
// Description: An optimized parallel sum reduction using shared memory.
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

// Use a power-of-two block size for simplicity
#define BLOCK_SIZE 256

void checkCudaError(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Optimized parallel reduction kernel
__global__ void reductionKernel(const float* g_in, float* g_out, int N) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    // Each thread loads one element from global memory to shared memory
    sdata[tid] = (i < N) ? g_in[i] : 0.0f;
    __syncthreads();

    // The reduction is performed in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes its partial sum to global memory
    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    }
}

int main() {
    int N = 1048576; // 2^20 elements
    size_t size = N * sizeof(float);
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::cout << "Performing parallel sum reduction for " << N << " elements." << std::endl;

    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f; // Each element is 1.0, so the sum should be N
    }
    std::vector<float> h_out_gpu(numBlocks);

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, numBlocks * sizeof(float));

    cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);

    std::cout << "Launching kernel with " << numBlocks << " blocks..." << std::endl;
    reductionKernel<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out_gpu.data(), d_out, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Perform the final reduction step on the CPU
    // (A second kernel launch could also do this)
    float gpu_sum = 0.0f;
    for (int i = 0; i < numBlocks; ++i) {
        gpu_sum += h_out_gpu[i];
    }
    
    // --- Verification ---
    double cpu_sum = 0.0; // Use double for precision on CPU
    for (int i = 0; i < N; ++i) {
        cpu_sum += h_in[i];
    }

    std::cout << "\n--- Final Sums ---" << std::endl;
    std::cout << "GPU Result: " << gpu_sum << std::endl;
    std::cout << "CPU Result: " << cpu_sum << std::endl;

    float diff = fabs(gpu_sum - (float)cpu_sum);
    if (diff < 1e-4) {
        std::cout << "\nVerification Successful!" << std::endl;
    } else {
        std::cout << "\nVerification FAILED! Difference: " << diff << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}