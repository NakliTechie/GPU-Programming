// File: arch_inspector.cu
// Description: Demonstrates thread hierarchy and control divergence using printf.
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Kernel that prints thread and block information.
__global__ void arch_inspector_kernel() {
    // Calculate the thread's unique global ID.
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Part 1: Basic Identification ---
    // Each thread prints its coordinates. The output order is non-deterministic.
    printf("[ID] Block: %d, Thread: %d, Global ID: %d\n",
           blockIdx.x, threadIdx.x, global_id);

    // --- Part 2: Demonstrating Divergence ---
    // This 'if' statement will cause threads in the same warp to diverge.
    if ((threadIdx.x % 2) == 0) {
        // This is a separate printf, its output will be interleaved with the "ODD" one.
        printf("[Divergence] Global ID %d is EVEN inside its block.\n", global_id);
    } else {
        printf("[Divergence] Global ID %d is ODD inside its block.\n", global_id);
    }
}

int main() {
    // Use a small grid to keep the printf output manageable.
    int numBlocks = 2;
    int threadsPerBlock = 8;

    std::cout << "Launching a " << numBlocks << "x" << threadsPerBlock << " grid." << std::endl;
    std::cout << "--- Kernel Output Begins ---" << std::endl;

    // Launch the kernel.
    arch_inspector_kernel<<<numBlocks, threadsPerBlock>>>();

    // cudaDeviceSynchronize() is CRUCIAL for printf.
    // It forces the CPU to wait for the GPU to finish EVERYTHING,
    // including flushing all printf buffers to the console.
    // Without this, main() would likely exit before the GPU prints anything.
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error after kernel launch: " << cudaGetErrorString(err) << std::endl;
    }

    std::cout << "--- Kernel Output Ends ---" << std::endl;

    // --- Verification ---
    // For this lab, "verification" is a manual inspection of the output.
    // Did all 16 threads (2 blocks * 8 threads) print their messages?
    std::cout << "\nVERIFICATION:" << std::endl;
    std::cout << "Please check the output above. You should see 16 [ID] messages " << std::endl;
    std::cout << "and 16 [Divergence] messages, one for each Global ID from 0 to 15." << std::endl;
    std::cout << "The order will be jumbled. This demonstrates parallel execution." << std::endl;
    std::cout << "\nProgram Finished Successfully." << std::endl;

    return 0;
}