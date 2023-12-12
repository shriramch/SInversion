#include "rgf2_cuda.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

int kernels_num_blocks, kernels_num_threads;

void kernel_init(int n) { kernels_num_blocks = kernels_num_threads = n; }

__global__ void matrixSubtractKernel(float *A, float *B, float *result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        result[index] = A[index] - B[index];
    }
}

void matrixSubtracter(float *A, float *B, float *result, int n) {
    matrixSubtractKernel<<<kernels_num_blocks, kernels_num_threads>>>(
        A, B, result, n);
}

__global__ void matrixAddKernel(float *A, float *B, float *result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        result[index] = A[index] + B[index];
    }
}

void matrixAdder(float *A, float *B, float *result, int n) {
    matrixAddKernel<<<kernels_num_blocks, kernels_num_threads>>>(A, B, result,
                                                                 n);
}

__global__ void matrixScaleKernel(float *A, float k, float *result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        result[index] = A[index] * k;
    }
}

void matrixScaler(float *A, float k, float *result, int n) {
    matrixScaleKernel<<<kernels_num_blocks, kernels_num_threads>>>(A, k, result,
                                                                   n);
}