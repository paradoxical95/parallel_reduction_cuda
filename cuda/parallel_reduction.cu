#include <iostream>
#include <cuda_runtime.h>
#include <chrono>  // For timing in C++

// Array size and threads per block
#define N 1000000000
#define THREADS_PER_BLOCK 256

// CUDA Kernel for parallel reduction (sum)
__global__ void reduceSum(int *input, int *output) {
    __shared__ int sharedData[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sharedData[tid] = input[i];
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// CUDA Kernel for parallel reduction (max)
__global__ void reduceMax(int *input, int *output) {
    __shared__ int sharedData[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sharedData[tid] = input[i];
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] = max(sharedData[tid], sharedData[tid + stride]);
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// CUDA Kernel for parallel reduction (min)
__global__ void reduceMin(int *input, int *output) {
    __shared__ int sharedData[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sharedData[tid] = input[i];
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] = min(sharedData[tid], sharedData[tid + stride]);
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

int main() {
    int *h_input, *h_output, *d_input, *d_output;
    int numBlocks = N / THREADS_PER_BLOCK;

    // Allocate host memory
    h_input = (int *)malloc(N * sizeof(int));
    h_output = (int *)malloc(numBlocks * sizeof(int));

    // Initialize input array with random values
    for (int i = 0; i < N; i++) {
        h_input[i] = rand() % 1000;  // Random values between 0 and 999
    }

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, numBlocks * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Measure execution time for CUDA kernels
    auto cuda_start = std::chrono::high_resolution_clock::now();

    // Launch sum kernel
    reduceSum<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output);

    // Copy partial results back to host
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Reduce final sum on the CPU
    int totalSum = 0;
    for (int i = 0; i < numBlocks; i++) {
        totalSum += h_output[i];
    }

    // Launch max kernel
    reduceMax<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output);
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    int maxVal = h_output[0];
    for (int i = 1; i < numBlocks; i++) {
        if (h_output[i] > maxVal) maxVal = h_output[i];
    }

    // Launch min kernel
    reduceMin<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output);
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    int minVal = h_output[0];
    for (int i = 1; i < numBlocks; i++) {
        if (h_output[i] < minVal) minVal = h_output[i];
    }

    auto cuda_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cuda_duration = cuda_end - cuda_start;

    std::cout << "CUDA Sum: " << totalSum << std::endl;
    std::cout << "CUDA Max: " << maxVal << std::endl;
    std::cout << "CUDA Min: " << minVal << std::endl;
    std::cout << "CUDA Execution Time: " << cuda_duration.count() << " seconds" << std::endl;

    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
