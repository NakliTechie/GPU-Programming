// File: radixSort.cu
// Description: A single-pass, single-block Radix Sort based on the LSB, using a parallel scan.
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <numeric>

#define BLOCK_SIZE 16 // Must be a power of two for this simple kernel

__global__ void radixSortLsbKernel(const int* in, int* out) {
    __shared__ int s_keys[BLOCK_SIZE];
    __shared__ int s_scan_data[BLOCK_SIZE]; // For scan intermediate values

    int tid = threadIdx.x;

    // 1. Load keys into shared memory
    s_keys[tid] = in[tid];
    __syncthreads();

    // 2. Extract the LSB and prepare for scan
    int bit = s_keys[tid] & 1;
    s_scan_data[tid] = bit;
    __syncthreads();

    // 3. Perform an EXCLUSIVE scan in shared memory (Kogge-Stone style)
    for (unsigned int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        int temp = 0;
        if (tid >= stride) {
            temp = s_scan_data[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            s_scan_data[tid] += temp;
        }
        __syncthreads();
    }
    
    // At this point, s_scan_data[tid] holds the INCLUSIVE scan of the bits.
    // We convert to exclusive by taking the value from the previous element.
    int ones_before = (tid > 0) ? s_scan_data[tid - 1] : 0;
    __syncthreads();

    // 4. Get total number of 1s from the last element of the inclusive scan
    int total_ones = s_scan_data[BLOCK_SIZE - 1];
    __syncthreads();

    // 5. Calculate destination address
    int total_zeros = BLOCK_SIZE - total_ones;
    int dest_idx;
    if (bit == 0) {
        dest_idx = tid - ones_before; // My rank among 0s
    } else {
        dest_idx = total_zeros + ones_before; // Start of 1s bucket + my rank among 1s
    }

    // 6. Write to destination
    out[dest_idx] = s_keys[tid];
}

int main() {
    int N = BLOCK_SIZE;
    size_t size = N * sizeof(int);

    std::vector<int> h_in(N), h_out_gpu(N), h_out_cpu(N);

    // Initialize with a mix of even and odd numbers
    for (int i = 0; i < N; ++i) { h_in[i] = (N - 1 - i); }

    std::cout << "Input: "; for(int v : h_in) std::cout << v << " "; std::cout << std::endl;

    int *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);

    radixSortLsbKernel<<<1, BLOCK_SIZE>>>(d_in, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out_gpu.data(), h_out_gpu.size() * sizeof(int), cudaMemcpyDeviceToHost);

    // --- Verification on CPU ---
    std::vector<int> zeros, ones;
    for (int val : h_in) {
        if ((val & 1) == 0) zeros.push_back(val);
        else ones.push_back(val);
    }
    std::copy(zeros.begin(), zeros.end(), h_out_cpu.begin());
    std::copy(ones.begin(), ones.end(), h_out_cpu.begin() + zeros.size());

    std::cout << "\n--- LSB Sort Results ---" << std::endl;
    std::cout << "GPU: "; for(int v : h_out_gpu) std::cout << v << " "; std::cout << std::endl;
    std::cout << "CPU: "; for(int v : h_out_cpu) std::cout << v << " "; std::cout << std::endl;

    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_out_gpu[i] != h_out_cpu[i]) {
            success = false;
            break;
        }
    }
    std::cout << (success ? "\nVerification Successful!" : "\nVerification FAILED!") << std::endl;

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}