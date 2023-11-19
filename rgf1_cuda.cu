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
__global__ void matrixInversionKernel(const float *A, float *result, int n, cublasHandle_t cublasHandle) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        result[index] = A[index] * k;
    }
    // Code inspired by this stackoverflow https://stackoverflow.com/questions/37731103/cublas-matrix-inverse-much-slower-than-matlab
    // NOTE as mentioned the function cublasSgetriBatched is for SMALL matrixes:
    //      "This function is intended to be used for matrices of small sizes where the launch overhead is a significant factor."
    // cit. https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-getribatched

    int* dLUPivots; // Pivoting array
    int* dLUInfo; // Device array to store inversion status
    int batchSize = 1; // Assuming a single matrix inversion

    CUDA_CALL(cudaMalloc(&dLUPivots, n * sizeof(int)), "Failed to allocate dLUPivots!");
    CUDA_CALL(cudaMalloc(&dLUInfo, sizeof(int)), "Failed to allocate dLUInfo!");

    CUBLAS_CALL(cublasSgetrfBatched(cublasHandle, n, &A, n, dLUPivots, dLUInfo, batchSize), "Failed to perform LU decomp operation!");
    CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!"); // TODO, not sure i need this sync the previous kernel

    CUBLAS_CALL(cublasSgetriBatched(cublasHandle, n, &A, n, dLUPivots, result, n, dLUInfo, batchSize), "Failed to perform Inverse operation!");
    CUDA_CALL(cudaDeviceSynchronize(), "Failed to synchronize after kernel call!"); // TODO, not sure i need this sync the previous kernel

    CUDA_CALL(cudaFree(dLUPivots), "Failed to free dLUPivots!");
    CUDA_CALL(cudaFree(dLUInfo), "Failed to free dLUInfo!");
}

// TODO, test this function
__global__ void matrixTransposeKernel(const float* A, float* result, int n, cublasHandle_t cublasHandle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Transpose A and store the result in 'result'
    cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, A, n, &beta, A, n, result, n);
}

// CUDA-accelerated rgf1sided function

void rgf1sided_cuda(Matrix &A, Matrix &G, bool sym_mat, bool save_off_diag) {
    int blockSize, matrixSize;
    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    

    // Initialize the handle used for cuBLAS
    cublasHandle_t cublasHandle;
    CUBLAS_CALL(cublasCreate(&cublasHandle), "Failed to initialize cuBLAS!");

    // TODO, check if we need the cuSolver
    // cusolverDnHandle_t cusolverH = NULL;
    // cusolverDnCreate(&cusolverH);

    // Allocate memory for matrices on the GPU
    float *A, *G;
    size_t size = matrixSize * matrixSize * sizeof(float);
    cudaMalloc(&A, size);
    cudaMalloc(&G, size);
    
    // Copy matrices from host to device
    cudaMemcpy(A, A.getMat(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(G, G.getMat(), size, cudaMemcpyHostToDevice);
    
    // Launch CUDA kernels for matrix operations

    // 0. Inverse of the first block
    // TODO, double check indexes (this is the case with all the functions calls)
    matrixInversionKernel<<<1, 1>>>(A, G, blockSize, cublasHandle);

    int nblocks = matrixSize / blockSize;
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
            (&(A.lodiag[(i - 1)*blockSize*blockSize]), &(G.mdiag[(i - 1)*blockSize*blockSize]), AGi, blockSize);
        matrixMultiplyKernel<<<kernels_num_blocks, kernels_num_threads>>>
            (AGi, &(A.updiag[(i - 1)*blockSize*blockSize]), AAi, blockSize);
        matrixSubtractKernel<<<kernels_num_blocks, kernels_num_threads>>>
            (&(A.mdiag[i*blockSize*blockSize]), AAi, AGi, blockSize);
        matrixInversionKernel<<<1, 1>>>
            (AGi, &(G.mdiag[i * blockSize * blockSize]), blockSize, cublasHandle);

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
            (&(G.mdiag[(i + 1) * blockSize * blockSize]), &(A.lodiag[i * blockSize * blockSize]), Glf1, blockSize);
        matrixMultiplyKernel<<<kernels_num_blocks, kernels_num_threads>>>
            (Glf1, &(G.mdiag[i * blockSize * blockSize]), Glf, blockSize);

        if (save_off_diag) {
            matrixScaleKernel<<<kernels_num_blocks, kernels_num_threads>>>
                (Glf, -1, &(G.lodiag[i * blockSize * blockSize]), blockSize);

            if (sym_mat) {
                matrixTransposeKernel<<<kernels_num_blocks, kernels_num_threads>>>
                    (&(G.lodiag[i * blockSize * blockSize]), &(G.updiag[i * blockSize * blockSize]), blockSize, cublasHandle);
            } else {
                float *Guf, *Guf1;
                cudaMalloc(&Guf, blockSizeBytes);
                cudaMalloc(&Guf1, blockSizeBytes);

                matrixMultiplyKernel<<<kernels_num_blocks, kernels_num_threads>>>
                    (&(A.updiag[i * blockSize * blockSize]), &(G.mdiag[(i + 1) * blockSize * blockSize]), Guf1, blockSize);
                matrixMultiplyKernel<<<kernels_num_blocks, kernels_num_threads>>>
                    (&(G.mdiag[i * blockSize * blockSize]), Guf1, Guf, blockSize);
                matrixScaleKernel<<<kernels_num_blocks, kernels_num_threads>>>
                    (Guf, -1, &(G.updiag[i * blockSize * blockSize]), blockSize);

                // Free temporary GPU memory
                cudaFree(Guf);
                cudaFree(Guf1);
            }
        }

        matrixMultiplyKernel<<<kernels_num_blocks, kernels_num_threads>>>
            (&(A.updiag[i * blockSize * blockSize]), Glf, Glf1, blockSize);
        matrixMultiplyKernel<<<kernels_num_blocks, kernels_num_threads>>>
            (&(G.mdiag[i * blockSize * blockSize]), Glf1, Glf, blockSize);
        matrixAddKernel<<<kernels_num_blocks, kernels_num_threads>>>
            (&(G.mdiag[i * blockSize * blockSize]), Glf, &(G.mdiag[i * blockSize * blockSize]), blockSize);

        // Free temporary GPU memory
        cudaFree(Glf);
        cudaFree(Glf1);
    }
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
    cudaMemcpy(A.getMat(), A, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(B.getMat(), B, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    // Destroy cuBLAS handle
    CUBLAS_CALL(cublasDestroy(cublasHandle), "Failed to destroy cuBLAS!");
}
