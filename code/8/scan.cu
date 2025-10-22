// File: scan.cu
// Description: A complete single-block inclusive scan using the Kogge-Stone algorithm.
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <numeric> // For std::partial_sum

// Block size must be a power of two for this simple version
#define BLOCK_SIZE 16

void checkCudaError(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Kogge-Stone Inclusive Scan Kernel (for a single block)
__global__ void koggeStoneScan(const float* in, float* out, int N) {
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;

    // Load data into shared memory
    if (tid < N) {
        sdata[tid] = in[tid];
    } else {
        sdata[tid] = 0.0f; // Pad with identity element
    }
    __syncthreads();

    // The "up-sweep" phase
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = 0;
        if (tid >= stride) {
            temp = sdata[tid] + sdata[tid - stride];
        }
        __syncthreads(); // Sync 1: Wait for all reads to complete

        if (tid >= stride) {
            sdata[tid] = temp;
        }
        __syncthreads(); // Sync 2: Wait for all writes to complete
    }

    // Write the result from shared memory to global memory
    if (tid < N) {
        out[tid] = sdata[tid];
    }
}

int main() {
    int N = BLOCK_SIZE;
    size_t size = N * sizeof(float);
    std::cout << "Performing parallel scan for " << N << " elements." << std::endl;

    std::vector<float> h_in(N);
    std::vector<float> h_out_gpu(N);
    std::vector<float> h_out_cpu(N);

    // Initialize with a simple pattern
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);

    // Launch with a single block
    koggeStoneScan<<<1, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out_gpu.data(), d_out, size, cudaMemcpyDeviceToHost);

    // --- Verification ---
    std::partial_sum(h_in.begin(), h_in.end(), h_out_cpu.begin());

    std::cout << "\n--- Scan Results ---" << std::endl;
    std::cout << "GPU: ";
    for (int i = 0; i < N; ++i) std::cout << h_out_gpu[i] << " ";
    std::cout << std::endl;

    std::cout << "CPU: ";
    for (int i = 0; i < N; ++i) std::cout << h_out_cpu[i] << " ";
    std::cout << std::endl;
    
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_out_gpu[i] - h_out_cpu[i]) > 1e-5) {
            success = false;
            break;
        }
    }
    std::cout << (success ? "\nVerification Successful!" : "\nVerification FAILED!") << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}