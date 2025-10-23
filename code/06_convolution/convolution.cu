// File: convolution.cu
// Description: Tiled 2D convolution using constant and shared memory.
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

// --- Kernel & Host Configuration ---
#define TILE_WIDTH 16
#define FILTER_RADIUS 1
#define FILTER_DIM (2 * FILTER_RADIUS + 1) // = 3 for a 3x3 filter
// The shared memory tile needs to be larger to hold the halo
#define BLOCK_WIDTH (TILE_WIDTH + 2 * FILTER_RADIUS) 

// --- Device Code ---
// The filter is stored in fast constant memory, accessible by all threads.
__constant__ float F[FILTER_DIM * FILTER_DIM];

__global__ void tiledConv2D(const float* In, float* Out, int width, int height) {
    // Shared memory tile for the input image patch (includes halo)
    __shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];

    // Thread's coordinates within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate the top-left corner of the input tile this block will load
    int start_row = blockIdx.y * TILE_WIDTH - FILTER_RADIUS;
    int start_col = blockIdx.x * TILE_WIDTH - FILTER_RADIUS;
    
    // Each thread loads one element from global memory into the shared memory tile
    int in_row = start_row + ty;
    int in_col = start_col + tx;
    
    if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
        Ns[ty][tx] = In[in_row * width + in_col];
    } else {
        Ns[ty][tx] = 0.0f; // Pad with zero for pixels outside the image
    }

    __syncthreads(); // Wait for all threads to finish loading the tile

    // Only "inner" threads (those not loading the halo) compute an output
    if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
        int out_row = blockIdx.y * TILE_WIDTH + ty;
        int out_col = blockIdx.x * TILE_WIDTH + tx;
        
        if (out_row < height && out_col < width) {
            float sum = 0.0f;
            for (int i = 0; i < FILTER_DIM; i++) {
                for (int j = 0; j < FILTER_DIM; j++) {
                    // Read from shared memory using local coordinates
                    sum += Ns[ty + i][tx + j] * F[i * FILTER_DIM + j];
                }
            }
            Out[out_row * width + out_col] = sum;
        }
    }
}

// --- Host Code ---
void cpuConv2D(const std::vector<float>& in, std::vector<float>& out, const float* filter, int W, int H) {
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            float sum = 0.0f;
            for (int fr = -FILTER_RADIUS; fr <= FILTER_RADIUS; ++fr) {
                for (int fc = -FILTER_RADIUS; fc <= FILTER_RADIUS; ++fc) {
                    int in_r = r + fr;
                    int in_c = c + fc;
                    if (in_r >= 0 && in_r < H && in_c >= 0 && in_c < W) {
                        sum += in[in_r * W + in_c] * filter[(fr + FILTER_RADIUS) * FILTER_DIM + (fc + FILTER_RADIUS)];
                    }
                }
            }
            out[r * W + c] = sum;
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
    int W = 64, H = 64;
    size_t size = W * H * sizeof(float);
    
    std::vector<float> h_in(W*H), h_out_gpu(W*H), h_out_cpu(W*H);
    float h_filter[FILTER_DIM * FILTER_DIM];

    // Initialize input image with a simple pattern
    for(int i = 0; i < W * H; ++i) h_in[i] = 1.0f;
    // Initialize filter (3x3 box blur, all weights are 1/9)
    for(int i = 0; i < FILTER_DIM * FILTER_DIM; ++i) h_filter[i] = 1.0f / 9.0f;
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, h_filter, sizeof(h_filter));

    dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 numBlocks((W + TILE_WIDTH - 1) / TILE_WIDTH, (H + TILE_WIDTH - 1) / TILE_WIDTH);
    
    std::cout << "Launching tiled convolution kernel..." << std::endl;
    tiledConv2D<<<numBlocks, threadsPerBlock>>>(d_in, d_out, W, H);
    cudaDeviceSynchronize();
    std::cout << "Kernel finished." << std::endl;

    cudaMemcpy(h_out_gpu.data(), d_out, size, cudaMemcpyDeviceToHost);

    std::cout << "Verifying against CPU implementation..." << std::endl;
    cpuConv2D(h_in, h_out_cpu, h_filter, W, H);
    
    printMatrix(h_out_gpu, W, "GPU Result");
    printMatrix(h_out_cpu, W, "CPU Verification Result");

    bool success = true;
    for (int i=0; i < W*H; ++i) {
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