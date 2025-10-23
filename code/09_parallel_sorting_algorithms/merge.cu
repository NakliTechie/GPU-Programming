// File: merge.cu
// Description: A simplified, single-block parallel merge using co-rank.
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm> // For std::merge and std::sort

void checkCudaError(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// A simple sequential merge to be run by each thread on its sub-problem.
// This runs on the DEVICE.
__device__ void sequential_merge(const float* a, int m, const float* b, int n, float* c) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (a[i] <= b[j]) {
            c[k++] = a[i++];
        } else {
            c[k++] = b[j++];
        }
    }
    while (i < m) c[k++] = a[i++];
    while (j < n) c[k++] = b[j++];
}

// This is a simplified binary search to find the rank of a value from A within B.
// Rank = number of elements in B smaller than val.
__device__ int rank_in_b(float val, const float* B, int n) {
    int low = 0, high = n;
    while(low < high) {
        int mid = low + (high - low) / 2;
        if (B[mid] < val) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

// Parallel Merge Kernel (for a single block)
__global__ void parallelMerge(const float* A, int m, const float* B, int n, float* C) {
    int tid = threadIdx.x;
    
    // This kernel has one thread per element of A
    if (tid < m) {
        // 1. Find the rank of A[tid] in B
        int j = rank_in_b(A[tid], B, n);
        // 2. The final position of A[tid] in C is its own index + its rank in B
        C[tid + j] = A[tid];
    }
    __syncthreads();
    
    // This kernel has one thread per element of B
    if (tid < n) {
        // 1. Find the rank of B[tid] in A
        int i = rank_in_b(B[tid], A, m);
        // 2. The final position of B[tid] in C is its own index + its rank in A
        C[tid + i] = B[tid];
    }
}

int main() {
    int M = 8, N = 8;
    int total_size = M + N;
    size_t a_size = M * sizeof(float);
    size_t b_size = N * sizeof(float);
    size_t c_size = total_size * sizeof(float);

    std::vector<float> h_A(M), h_B(N), h_C_gpu(total_size), h_C_cpu(total_size);

    // Initialize with sorted data
    for (int i = 0; i < M; ++i) h_A[i] = i * 2;       // A = {0, 2, 4, ...}
    for (int i = 0; i < N; ++i) h_B[i] = i * 2 + 1;   // B = {1, 3, 5, ...}
    
    std::cout << "Input A: "; for(float v : h_A) std::cout << v << " "; std::cout << std::endl;
    std::cout << "Input B: "; for(float v : h_B) std::cout << v << " "; std::cout << std::endl;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, a_size);
    cudaMalloc(&d_B, b_size);
    cudaMalloc(&d_C, c_size);

    cudaMemcpy(d_A, h_A.data(), a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), b_size, cudaMemcpyHostToDevice);
    
    // Launch with one block, enough threads to cover the larger of the two inputs
    parallelMerge<<<1, std::max(M,N)>>>(d_A, M, d_B, N, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_gpu.data(), d_C, c_size, cudaMemcpyDeviceToHost);

    // --- Verification ---
    std::merge(h_A.begin(), h_A.end(), h_B.begin(), h_B.end(), h_C_cpu.begin());

    std::cout << "\n--- Merge Results ---" << std::endl;
    std::cout << "GPU: "; for(float v : h_C_gpu) std::cout << v << " "; std::cout << std::endl;
    std::cout << "CPU: "; for(float v : h_C_cpu) std::cout << v << " "; std::cout << std::endl;
    
    bool success = true;
    for (int i = 0; i < total_size; ++i) {
        if (fabs(h_C_gpu[i] - h_C_cpu[i]) > 1e-5) {
            success = false;
            break;
        }
    }
    std::cout << (success ? "\nVerification Successful!" : "\nVerification FAILED!") << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}