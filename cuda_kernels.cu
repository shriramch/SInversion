#include "rgf2_cuda.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

int kernels_num_blocks, kernels_num_threads;

// void kernel_init(int n) { kernels_num_blocks = kernels_num_threads = n; }
void kernel_init(int n) {
    kernels_num_threads = 1024;
    kernels_num_blocks =
        (n * n + kernels_num_threads - 1) / kernels_num_threads;
}

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

__global__ void matrixMultiplyKernel_old(float *A, float *B, float *result,
                                         int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        result[row * n + col] = sum;
    }
}

__global__ void setIdentityMatrixKernel(float *result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        int i = index / n;
        int j = index % n;
        result[index] = (i == j) ? 1 : 0;
    }
}

void setIdentityMatrix(float *result, int n) {
    setIdentityMatrixKernel<<<kernels_num_blocks, kernels_num_threads>>>(result,
                                                                         n);
}
