#include <cuda_runtime.h>
#include "rgf1_cuda.hpp"
#include <cublas_v2.h>
#include <cusolverDn.h>


// CUDA Kernel for a specific matrix operation


__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void matrixSubtractKernel(float *A, float *B, float *C, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        C[index] = A[index] - B[index];
    }
}

__global__ void matrixAddKernel(float *A, float *B, float *C, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        C[index] = A[index] + B[index];
    }
}

__global__ void matrixScaleKernel(float *A, float k, float *B, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        B[index] = A[index] * k;
    }
}

// CUDA-accelerated rgf1sided function

void rgf1sided_cuda(Matrix &A, Matrix &G, bool sym_mat, bool save_off_diag) {
    int blockSize, matrixSize;
    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    
    // Matrix Inversion CUDA
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;

    // Initialize cuSOLVER and cuBLAS
    cusolverDnCreate(&cusolverH);
    cublasCreate(&cublasH);

    // Allocate memory for matrices on the GPU
    float *d_A, *d_B, *d_C; // Example matrices
    size_t size = matrixSize * matrixSize * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
  /*  
    // 0. Inverse of the first block
    A.invBLAS(blockSize, A.mdiag, G.mdiag);

    int nblocks = matrixSize / blockSize;

 

    // Copy matrices from host to device
    cudaMemcpy(d_A, A.getMat(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.getMat(), size, cudaMemcpyHostToDevice);

    
    // 1. Forward substitution (performed left to right)
    for (int i = 1; i < nblocks; ++i) {
        float *AAi = new float[blockSize * blockSize](), *AGi = new float[blockSize * blockSize]();
        A.mmmBLAS(blockSize, &(A.lodiag[(i - 1)*blockSize*blockSize]), &(G.mdiag[(i - 1)*blockSize*blockSize]), AGi);
        A.mmmBLAS(blockSize, AGi, &(A.updiag[(i - 1)*blockSize*blockSize]), AAi);
        A.mmSub(blockSize, &(A.mdiag[i*blockSize*blockSize]), AAi, AGi);
        A.invBLAS(blockSize, AGi, &(G.mdiag[i*blockSize*blockSize]));

        delete [] AAi;
        delete [] AGi;
    }

    // 2. Backfward 
    for (int i = nblocks - 2; i >= 0; --i) {
        // float *g_ii = G.mdiag + i*blockSize*blockSize;
        float *Glf = new float[nblocks * nblocks], *Glf1 = new float[nblocks * nblocks];
        A.mmmBLAS(blockSize, &(G.mdiag[(i + 1)*blockSize*blockSize]), &(A.lodiag[i*blockSize*blockSize]), Glf1);
        A.mmmBLAS(blockSize, Glf1, &(G.mdiag[i*blockSize*blockSize]), Glf);

        if (save_off_diag) {
            A.matScale(blockSize, Glf, -1, &(G.lodiag[i*blockSize*blockSize]));
            if (sym_mat) {
                A.transposeBLAS(blockSize, &(G.lodiag[i*blockSize*blockSize]), &(G.updiag[i*blockSize*blockSize]));
            } else {
                float *Guf = new float[blockSize * blockSize], *Guf1 = new float[blockSize * blockSize];
                A.mmmBLAS(blockSize, &(A.updiag[i*blockSize*blockSize]), &(G.mdiag[(i + 1)*blockSize*blockSize]), Guf1);
                A.mmmBLAS(blockSize, &(G.mdiag[i*blockSize*blockSize]), Guf1, Guf);
                A.matScale(blockSize, Guf, -1, &(G.updiag[i*blockSize*blockSize]));

                delete[] Guf;
                delete[] Guf1;
            }
        }

        A.mmmBLAS(blockSize, &(A.updiag[i*blockSize*blockSize]), Glf, Glf1);
        A.mmmBLAS(blockSize, &(G.mdiag[i*blockSize*blockSize]), Glf1, Glf);
        A.mmAdd(blockSize, &(G.mdiag[i*blockSize*blockSize]), Glf, &(G.mdiag[i*blockSize*blockSize]));

        delete [] Glf;
        delete[] Glf1;
    }
    */

    // Copy results back to host
    cudaMemcpy(A.getMat(), d_A, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(B.getMat(), d_B, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
