// File: stencil3d.cu
// Description: A tiled 3D stencil computation using shared memory with halos.
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

// --- Configuration ---
#define N 32 // Grid dimension (must be multiple of TILE_DIM) - Reduced for faster simulation
#define TILE_DIM 8 // Reduced tile size
#define RADIUS 1
#define BLOCK_DIM (TILE_DIM + 2 * RADIUS) // Shared memory tile size

void checkCudaError(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Tiled 3D 7-point Stencil Kernel
__global__ void tiledStencil3D(const float* in, float* out) {
    __shared__ float tile[BLOCK_DIM][BLOCK_DIM][BLOCK_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // Global indices for loading data - load all elements in Z dimension for this thread
    for (int z_iter = tz; z_iter < BLOCK_DIM; z_iter += blockDim.z) {
        int iz = blockIdx.z * TILE_DIM + z_iter - RADIUS;
        int ix = blockIdx.x * TILE_DIM + tx - RADIUS;
        int iy = blockIdx.y * TILE_DIM + ty - RADIUS;

        // Load tile from global to shared memory, handling boundaries
        if (ix >= 0 && ix < N && iy >= 0 && iy < N && iz >= 0 && iz < N) {
            tile[z_iter][ty][tx] = in[iz * N * N + iy * N + ix];
        } else {
            tile[z_iter][ty][tx] = 0.0f;
        }
    }

    __syncthreads();

    // Only "inner" threads compute output values
    // Process multiple z-dimension outputs for this thread
    for (int z_iter = tz + RADIUS; z_iter < BLOCK_DIM - RADIUS; z_iter += blockDim.z) {
        if (tx >= RADIUS && tx < BLOCK_DIM - RADIUS &&
            ty >= RADIUS && ty < BLOCK_DIM - RADIUS)
        {
            int out_x = blockIdx.x * TILE_DIM + (tx - RADIUS);
            int out_y = blockIdx.y * TILE_DIM + (ty - RADIUS);
            int out_z = blockIdx.z * TILE_DIM + (z_iter - RADIUS);

            if (out_x < N && out_y < N && out_z < N) {
                float center = tile[z_iter][ty][tx];
                float up     = tile[z_iter + 1][ty][tx];
                float down   = tile[z_iter - 1][ty][tx];
                float north  = tile[z_iter][ty + 1][tx];
                float south  = tile[z_iter][ty - 1][tx];
                float east   = tile[z_iter][ty][tx + 1];
                float west   = tile[z_iter][ty][tx - 1];

                out[out_z * N * N + out_y * N + out_x] = 0.5f * center + 0.5f * (up + down + north + south + east + west) / 6.0f;
            }
        }
    }
}

// Host function for CPU verification
void cpuStencil3D(const std::vector<float>& in, std::vector<float>& out) {
    for (int z = 0; z < N; ++z) {
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                float sum = 0.0f;
                // Check boundaries for each neighbor
                if (x > 0) sum += in[z*N*N + y*N + (x-1)];
                if (x < N-1) sum += in[z*N*N + y*N + (x+1)];
                if (y > 0) sum += in[z*N*N + (y-1)*N + x];
                if (y < N-1) sum += in[z*N*N + (y+1)*N + x];
                if (z > 0) sum += in[(z-1)*N*N + y*N + x];
                if (z < N-1) sum += in[(z+1)*N*N + y*N + x];

                out[z*N*N + y*N + x] = 0.5f * in[z*N*N + y*N + x] + 0.5f * sum / 6.0f;
            }
        }
    }
}

int main() {
    size_t size = N * N * N * sizeof(float);
    std::cout << "Tiled 3D Stencil for a " << N << "x" << N << "x" << N << " grid." << std::endl;

    std::vector<float> h_in(N*N*N), h_out_gpu(N*N*N), h_out_cpu(N*N*N);

    // Initialize with a "hot spot" in the middle
    for(int i = 0; i < N*N*N; ++i) h_in[i] = 0.0f;
    h_in[(N/2)*N*N + (N/2)*N + (N/2)] = 100.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, size); cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM / 8); // Use smaller Z-dim for blocks
    dim3 numBlocks(N / TILE_DIM, N / TILE_DIM, N / TILE_DIM);

    tiledStencil3D<<<numBlocks, threadsPerBlock>>>(d_in, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out_gpu.data(), d_out, size, cudaMemcpyDeviceToHost);

    cpuStencil3D(h_in, h_out_cpu);

    int center_idx = (N/2)*N*N + (N/2)*N + (N/2);
    std::cout << "\n--- Center Voxel Value ---" << std::endl;
    std::cout << "GPU Result: " << h_out_gpu[center_idx] << std::endl;
    std::cout << "CPU Result: " << h_out_cpu[center_idx] << std::endl;
    
    bool success = true;
    for(int i=0; i<N*N*N; ++i) {
        if (fabs(h_out_gpu[i] - h_out_cpu[i]) > 1e-4) {
            success = false; break;
        }
    }
    std::cout << (success ? "\nVerification Successful!" : "\nVerification FAILED!") << std::endl;

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}