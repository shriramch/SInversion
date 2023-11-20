#include <cuda_runtime.h>
#include "rgf1_cuda.hpp" // TODO, uncomment once figured DaVinci sudo problem
#include <cublas_v2.h>
#include <cusolverDn.h>

// CUDA Kernel for a specific matrix operation
// Note, thanks to dynamic parallelism on cuda 5.0 and over, it IS possible to launch a kernel inside another kernel


__global__ void matrixMultiplyKernel(float *A, float *B, float *result, int n) {
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

__global__ void matrixSubtractKernel(float *A, float *B, float *result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        result[index] = A[index] - B[index];
    }
}

__global__ void matrixAddKernel(float *A, float *B, float *result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        result[index] = A[index] + B[index];
    }
}

__global__ void matrixScaleKernel(float *A, float k, float *result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        result[index] = A[index] * k;
    }
}

// TODO, test this function
// TODO, after testing you can also delete all the messages if you want
// TODO, note this is declared as a kernel, but the function cublasSgetrfBatched is already a kernel, so it would make
//      more sense to declare this as just a C++ function (without the global) 
__global__ void matrixInversionKernel(float *A, float *result, int n, cublasHandle_t cublasHandle) {
    // Code inspired by this stackoverflow https://stackoverflow.com/questions/37731103/cublas-matrix-inverse-much-slower-than-matlab
    // NOTE as mentioned the function cublasSgetriBatched is for SMALL matrixes:
    //      "This function is intended to be used for matrices of small sizes where the launch overhead is a significant factor."
    // cit. https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-getribatched

    int* dLUPivots; // Pivoting array
    int* dLUInfo; // Device array to store inversion status
    int batchSize = 1; // Assuming a single matrix inversion

    cudaMalloc(&dLUPivots, n * sizeof(int));
    cudaMalloc(&dLUInfo, sizeof(int));

    cublasSgetrfBatched(cublasHandle, n, &A, n, dLUPivots, dLUInfo, batchSize); // RIP does not work, NOTE it expect a float** NOT a float*
    cudaDeviceSynchronize(); // TODO, not sure i need this sync the previous kernel

    cublasSgetriBatched(cublasHandle, n, &A, n, dLUPivots, &result, n, dLUInfo, batchSize);
    cudaDeviceSynchronize(); // TODO, not sure i need this sync the previous kernel

    cudaFree(dLUPivots);
    cudaFree(dLUInfo);
}

// TODO, test this function
__global__ void matrixTransposeKernel(const float* A, float* result, int n, cublasHandle_t cublasHandle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Transpose A and store the result in 'result'
    cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, A, n, &beta, A, n, result, n);
}

// CUDA-accelerated rgf1sided function

void rgf1sided_cuda(Matrix &input_A, Matrix &input_G, bool sym_mat, bool save_off_diag) {
    int blockSize, matrixSize;
    input_A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    int nblocks = matrixSize / blockSize;
    

    // Initialize the handle used for cuBLAS
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    // TODO, check if we need the cuSolver
    // cusolverDnHandle_t cusolverH = NULL;
    // cusolverDnCreate(&cusolverH);

    // Allocate memory for matrices on the GPU
    float *A, *G;
    size_t size = matrixSize * matrixSize * sizeof(float);
    cudaMalloc(&A, size);
    cudaMalloc(&G, size);
    
    // Copy matrices from host to device
    cudaMemcpy(A, input_A.getMat(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(G, input_G.getMat(), size, cudaMemcpyHostToDevice);

    // TODO, prob not the best optimized way to do this calculation
    // Allocate memory for Matrix specifics on the GPU
    float *A_mdiag, *G_mdiag;
    size_t size_mdiag = nblocks * blockSize * blockSize * sizeof(float);
    cudaMalloc(&A_mdiag, size_mdiag);
    cudaMalloc(&G_mdiag, size_mdiag);
    // Copy matrices from host to device
    cudaMemcpy(A_mdiag, input_A.mdiag, size_mdiag, cudaMemcpyHostToDevice);
    cudaMemcpy(G_mdiag, input_G.mdiag, size_mdiag, cudaMemcpyHostToDevice);
    
    float *A_updiag, *G_updiag;
    size_t size_updiag = (nblocks - 1) * blockSize * blockSize * sizeof(float);
    cudaMalloc(&A_updiag, size_updiag);
    cudaMalloc(&G_updiag, size_updiag);
    // Copy matrices from host to device
    cudaMemcpy(A_updiag, input_A.mdiag, size_updiag, cudaMemcpyHostToDevice);
    cudaMemcpy(G_updiag, input_G.mdiag, size_updiag, cudaMemcpyHostToDevice);

    float *A_lodiag, *G_lodiag;
    // size_t size_lodiag = (nblocks - 1) * blockSize * blockSize * sizeof(float); // = size_updiag
    cudaMalloc(&A_lodiag, size_updiag);
    cudaMalloc(&G_lodiag, size_updiag);
    // Copy matrices from host to device
    cudaMemcpy(A_lodiag, input_A.mdiag, size_updiag, cudaMemcpyHostToDevice);
    cudaMemcpy(G_lodiag, input_G.mdiag, size_updiag, cudaMemcpyHostToDevice);
    
    // Launch CUDA kernels for matrix operations

    // 0. Inverse of the first block
    // TODO, double check indexes (this is the case with all the functions calls)
    matrixInversionKernel<<<1, 1>>>(A, G, blockSize, cublasHandle);

    int kernels_num_blocks = nblocks; // TODO, find the optimal combination
    int kernels_num_threads = 1; // TODO, find the optimal combination
    
    // 1. Forward substitution (performed left to right)
    for (int i = 1; i < nblocks; ++i) {
        float *AAi, *AGi;
        size_t blockSizeBytes = blockSize * blockSize * sizeof(float);
        cudaMalloc(&AAi, blockSizeBytes);
        cudaMalloc(&AGi, blockSizeBytes);

        // TODO, check how to parallelize, since u need the previous G
        matrixMultiplyKernel<<<kernels_num_blocks, kernels_num_threads>>>
            (&(A_lodiag[(i - 1)*blockSize*blockSize]), &(G_mdiag[(i - 1)*blockSize*blockSize]), AGi, blockSize);
        matrixMultiplyKernel<<<kernels_num_blocks, kernels_num_threads>>>
            (AGi, &(A_updiag[(i - 1)*blockSize*blockSize]), AAi, blockSize);
        matrixSubtractKernel<<<kernels_num_blocks, kernels_num_threads>>>
            (&(A_mdiag[i*blockSize*blockSize]), AAi, AGi, blockSize);
        matrixInversionKernel<<<1, 1>>>
            (AGi, &(G_mdiag[i * blockSize * blockSize]), blockSize, cublasHandle);

        // Free temporary GPU memory
        cudaFree(AAi);
        cudaFree(AGi);
    }

    // 2. Backward substitution
    for (int i = nblocks - 2; i >= 0; --i) {
        float *Glf, *Glf1;
        size_t blockSizeBytes = blockSize * blockSize * sizeof(float);
        cudaMalloc(&Glf, blockSizeBytes);
        cudaMalloc(&Glf1, blockSizeBytes);

        matrixMultiplyKernel<<<kernels_num_blocks, kernels_num_threads>>>
            (&(G_mdiag[(i + 1) * blockSize * blockSize]), &(A_lodiag[i * blockSize * blockSize]), Glf1, blockSize);
        matrixMultiplyKernel<<<kernels_num_blocks, kernels_num_threads>>>
            (Glf1, &(G_mdiag[i * blockSize * blockSize]), Glf, blockSize);

        if (save_off_diag) {
            matrixScaleKernel<<<kernels_num_blocks, kernels_num_threads>>>
                (Glf, -1, &(G_lodiag[i * blockSize * blockSize]), blockSize);

            if (sym_mat) {
                matrixTransposeKernel<<<kernels_num_blocks, kernels_num_threads>>>
                    (&(G_lodiag[i * blockSize * blockSize]), &(G_updiag[i * blockSize * blockSize]), blockSize, cublasHandle);
            } else {
                float *Guf, *Guf1;
                cudaMalloc(&Guf, blockSizeBytes);
                cudaMalloc(&Guf1, blockSizeBytes);

                matrixMultiplyKernel<<<kernels_num_blocks, kernels_num_threads>>>
                    (&(A_updiag[i * blockSize * blockSize]), &(G_mdiag[(i + 1) * blockSize * blockSize]), Guf1, blockSize);
                matrixMultiplyKernel<<<kernels_num_blocks, kernels_num_threads>>>
                    (&(G_mdiag[i * blockSize * blockSize]), Guf1, Guf, blockSize);
                matrixScaleKernel<<<kernels_num_blocks, kernels_num_threads>>>
                    (Guf, -1, &(G_updiag[i * blockSize * blockSize]), blockSize);

                // Free temporary GPU memory
                cudaFree(Guf);
                cudaFree(Guf1);
            }
        }

        matrixMultiplyKernel<<<kernels_num_blocks, kernels_num_threads>>>
            (&(A_updiag[i * blockSize * blockSize]), Glf, Glf1, blockSize);
        matrixMultiplyKernel<<<kernels_num_blocks, kernels_num_threads>>>
            (&(G_mdiag[i * blockSize * blockSize]), Glf1, Glf, blockSize);
        matrixAddKernel<<<kernels_num_blocks, kernels_num_threads>>>
            (&(G_mdiag[i * blockSize * blockSize]), Glf, &(G_mdiag[i * blockSize * blockSize]), blockSize);

        // Free temporary GPU memory
        cudaFree(Glf);
        cudaFree(Glf1);
    }

    // Copy results back to host
    cudaMemcpy(input_A.getMat(), A, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.getMat(), G, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_A.mdiag, A_mdiag, size_mdiag, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.mdiag, G_mdiag, size_mdiag, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_A.updiag, A_updiag, size_updiag, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.updiag, G_updiag, size_updiag, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_A.lodiag, A_lodiag, size_updiag, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.lodiag, G_lodiag, size_updiag, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(A);
    cudaFree(G);
    cudaFree(A_mdiag);
    cudaFree(G_mdiag);
    cudaFree(A_updiag);
    cudaFree(G_updiag);
    cudaFree(A_lodiag);
    cudaFree(G_lodiag);

    // Destroy cuBLAS handle
    cublasDestroy(cublasHandle);
}
