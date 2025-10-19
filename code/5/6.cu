// File: coalescedMatMul.cu
// Description: An optimized tiled matrix multiplication with coalesced global memory access.
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_WIDTH 16

void checkCudaError(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Tiled Matrix Multiplication Kernel with COALESCED memory access
__global__ void coalescedTiledMatMul(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;

    for (int t = 0; t < N / TILE_WIDTH; t++) {
        // --- OPTIMIZED LOADING ---
        // 1. Load for As is coalesced because adjacent threads (tx=0,1,2..)
        //    load adjacent columns from the same row.
        As[ty][tx] = A[row * N + (t * TILE_WIDTH + tx)];

        // 2. Load for Bs is also coalesced. Adjacent threads (tx=0,1,2..)
        //    load adjacent columns. We store it transposed in shared memory.
        Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];

        __syncthreads();

        // Multiply the tiles from shared memory.
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
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
    int N = 64;
    if (N % TILE_WIDTH != 0) { std::cerr << "Matrix size must be a multiple of TILE_WIDTH." << std::endl; return -1; }
    size_t size = N * N * sizeof(float);
    std::cout << "Coalesced tiled matrix multiplication for " << N << "x" << N << " matrices." << std::endl;

    std::vector<float> h_A(N * N), h_B(N * N), h_C_gpu(N * N), h_C_cpu(N * N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = (float)i;
            h_B[i * N + j] = (float)j;
        }
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size); cudaMalloc(&d_B, size); cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks(N / TILE_WIDTH, N / TILE_WIDTH);

    coalescedTiledMatMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_gpu.data(), d_C, size, cudaMemcpyDeviceToHost);

    cpuMatMul(h_A, h_B, h_C_cpu, N);

    printMatrix(h_C_gpu, N, "GPU Result");
    printMatrix(h_C_cpu, N, "CPU Verification Result");

    bool success = true;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C_gpu[i] - h_C_cpu[i]) > 0.1) {
            success = false;
            break;
        }
    }
    std::cout << (success ? "\nVerification Successful!" : "\nVerification FAILED!") << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}