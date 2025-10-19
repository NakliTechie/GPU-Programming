// File: matMul.cu
// Description: A complete, verifiable, naive implementation of matrix multiplication.
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

// Helper for checking CUDA API errors
void checkCudaError(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Naive Matrix Multiplication Kernel
__global__ void matMul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Host function for CPU-based verification
void cpuMatMul(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// Host function to print a small corner of a matrix
void printMatrix(const std::vector<float>& M, int N, const std::string& name) {
    std::cout << "--- Top-left 3x3 of " << name << " ---" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << M[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }
}

int main() {
    int N = 256; // Use a size divisible by block size for simplicity
    size_t size = N * N * sizeof(float);
    std::cout << "Performing matrix multiplication for " << N << "x" << N << " matrices." << std::endl;

    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C_gpu(N * N);
    std::vector<float> h_C_cpu(N * N);

    // Initialize host matrices: A[r,c] = r, B[r,c] = c
    // The result C[r,c] will have a predictable pattern.
    for (int i = 0; i < N; ++i) { // row
        for (int j = 0; j < N; ++j) { // col
            h_A[i * N + j] = (float)i;
            h_B[i * N + j] = (float)j;
        }
    }

    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, size), "Malloc A");
    checkCudaError(cudaMalloc(&d_B, size), "Malloc B");
    checkCudaError(cudaMalloc(&d_C, size), "Malloc C");

    checkCudaError(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice), "Memcpy A");
    checkCudaError(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice), "Memcpy B");

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    matMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    checkCudaError(cudaDeviceSynchronize(), "Kernel launch/sync");

    checkCudaError(cudaMemcpy(h_C_gpu.data(), d_C, size, cudaMemcpyDeviceToHost), "Memcpy C");

    // --- Verification ---
    cpuMatMul(h_A, h_B, h_C_cpu, N);

    printMatrix(h_C_gpu, N, "GPU Result");
    printMatrix(h_C_cpu, N, "CPU Verification Result");

    bool success = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C_gpu[i] - h_C_cpu[i]) > 0.1) { // A generous tolerance for floats
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "\nVerification Successful!" << std::endl;
    } else {
        std::cout << "\nVerification FAILED!" << std::endl;
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}