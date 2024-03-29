#include "rgf1_cuda.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

float *identity_matrix;
int *d_info = nullptr;
float *d_A, *d_identity, *d_work;
int *ipiv;
float *d_result;

// Function to multiply two matrices on the GPU
void matrixMultiplyKernel(float *A, float *B, float *result, int n,
                          cublasHandle_t cublasHandle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, B, n,
                A, n, &beta, result, n);
}

__global__ void mulmul(float *A, float *B, float *result, int n) {
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

__global__ void setIdentityMatrix(float *result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        int i = index / n;
        int j = index % n;
        result[index] = (i == j) ? 1 : 0;
    }
}

// Function to create an identity matrix of size n x n
float *createIdentityMatrix(int n) {
    float *identityMatrix = (float *)malloc(n * n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int index = i * n + j;
            identityMatrix[index] = (i == j) ? 1 : 0;
        }
    }
    return identityMatrix;
}

// Function to invert a matrix on the GPU using cuSolver
void matrixInversionKernel(float *A, float *result, int n,
                           cusolverDnHandle_t cusolverHandle) {

    // Copy the input matrix A to the device
    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyDeviceToDevice);

    int num_threads = 1024;
    int num_blocks = (n * n + num_threads - 1) / num_threads;
    setIdentityMatrix<<<num_blocks, num_threads>>>(d_identity, n);

    // Perform LU decomposition on the device
    cusolverDnSgetrf(cusolverHandle, n, n, d_A, n, d_work, NULL,
                     d_info); // Not using PIVOT for now

    // Solving AX = I  , where X is the result_matrix, and I is the
    // identity_matrix. Since AA^(-1) = I It saves on the result_matrix
    // (identity) the answer
    cusolverDnSgetrs(cusolverHandle, CUBLAS_OP_N, n, n, d_A, n, NULL,
                     d_identity, n, d_info); // Not using PIVOT for now

    cudaMemcpy(result, d_identity, n * n * sizeof(float),
               cudaMemcpyDeviceToDevice);
}

// Function to transpose a matrix on the GPU using cuBLAS
void matrixTransposeKernel(const float *A, float *result, int n,
                           cublasHandle_t cublasHandle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform the transposition
    cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, A, n,
                &beta, NULL, n, result, n);
}

/**
 * @brief Performs a one-sided RGF inversion on a given matrix using CUDA.
 *
 * This function performs a one-sided RGF inversion on a given matrix using
 * CUDA.
 *
 * @param A The matrix on which the RGF inversion is to be performed.
 * @param G The matrix that will hold the result of the RGF inversion.
 * @param sym_mat A boolean flag indicating whether the input matrix is
 * symmetric. Default is false.
 * @param save_off_diag A boolean flag indicating whether to save the
 * off-diagonal elements of the matrix. Default is true.
 *
 * @return void
 *
 * @note This function assumes that the size of the matrix is divisible by the
 * block size.
 */
void rgf1sided_cuda(Matrix &input_A, Matrix &input_G, bool sym_mat,
                    bool save_off_diag) {
    int blockSize, matrixSize;
    input_A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    int nblocks = matrixSize / blockSize;

    // int kernels_num_blocks = nblocks;
    // int kernels_num_threads = nblocks;
    int kernels_num_threads = 1024; // Max threads per thread-block
    int kernels_num_blocks =
        (nblocks * nblocks + kernels_num_threads - 1) / kernels_num_threads;
    size_t blockSizeBytes = blockSize * blockSize * sizeof(float);

    // Initialize the handle used for cuBLAS
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;

    cublasCreate(&cublasHandle);
    cusolverDnCreate(&cusolverHandle);

    // Allocate memory for Matrix specifics on the GPU
    size_t size_mdiag = nblocks * blockSize * blockSize * sizeof(float);
    size_t size_updiag = (nblocks - 1) * blockSize * blockSize * sizeof(float);
    float *A_mat, *A_mdiag, *G_mat, *G_mdiag;
    float *A_updiag, *G_updiag;
    float *A_lodiag, *G_lodiag;
    cudaMalloc(&A_mat, size_mdiag + 2 * size_updiag + 6 * blockSizeBytes);
    cudaMalloc(&G_mat, size_mdiag + 2 * size_updiag);
    A_mdiag = A_mat;
    A_updiag = A_mat + size_mdiag / sizeof(float);
    A_lodiag = A_updiag + size_updiag / sizeof(float);
    G_mdiag = G_mat;
    G_updiag = G_mat + size_mdiag / sizeof(float);
    G_lodiag = G_updiag + size_updiag / sizeof(float);

    // Inverse and transpose kernel variables
    cudaMalloc(&d_info, sizeof(int));
    cudaMalloc(&d_A, blockSize * blockSize * sizeof(float));
    cudaMalloc(&d_identity, blockSize * blockSize * sizeof(float));
    cudaMalloc(&d_work, blockSize * blockSize * sizeof(float));
    cudaMalloc(&ipiv, blockSize * sizeof(int));
    cudaMalloc((void **)&d_result, blockSize * blockSize * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(A_mdiag, input_A.mdiag, size_mdiag, cudaMemcpyHostToDevice);
    cudaMemcpy(A_updiag, input_A.updiag, size_updiag, cudaMemcpyHostToDevice);
    cudaMemcpy(A_lodiag, input_A.lodiag, size_updiag, cudaMemcpyHostToDevice);

    // Utility matrices
    float *AAi, *AGi;
    AAi = A_lodiag + size_updiag / sizeof(float);
    AGi = AAi + blockSizeBytes / sizeof(float);
    float *Glf, *Glf1;
    Glf = AGi + blockSizeBytes / sizeof(float);
    Glf1 = Glf + blockSizeBytes / sizeof(float);
    float *Guf, *Guf1;
    Guf = Glf1 + blockSizeBytes / sizeof(float);
    Guf1 = Guf + blockSizeBytes / sizeof(float);

    // identity_matrix = createIdentityMatrix(blockSize);

    // Launch CUDA kernels for matrix operations

    // 0. Inverse of the first block
    matrixInversionKernel(A_mdiag, G_mdiag, blockSize, cusolverHandle);
    // 1. Forward substitution (performed left to right)

    for (int i = 1; i < nblocks; ++i) {
        matrixMultiplyKernel(&(A_lodiag[(i - 1) * blockSize * blockSize]),
                             &(G_mdiag[(i - 1) * blockSize * blockSize]), AGi,
                             blockSize, cublasHandle);

        matrixMultiplyKernel(AGi, &(A_updiag[(i - 1) * blockSize * blockSize]),
                             AAi, blockSize, cublasHandle);

        matrixSubtractKernel<<<kernels_num_blocks, kernels_num_threads>>>(
            &(A_mdiag[i * blockSize * blockSize]), AAi, AGi, blockSize);

        matrixInversionKernel(AGi, &(G_mdiag[i * blockSize * blockSize]),
                              blockSize, cusolverHandle);
    }

    for (int i = nblocks - 2; i >= 0; --i) {
        matrixMultiplyKernel(&(G_mdiag[(i + 1) * blockSize * blockSize]),
                             &(A_lodiag[i * blockSize * blockSize]), Glf1,
                             blockSize, cublasHandle);
        matrixMultiplyKernel(Glf1, &(G_mdiag[i * blockSize * blockSize]), Glf,
                             blockSize, cublasHandle);

        if (save_off_diag) {
            matrixScaleKernel<<<kernels_num_blocks, kernels_num_threads>>>(
                Glf, -1, &(G_lodiag[i * blockSize * blockSize]), blockSize);

            if (sym_mat) {
                matrixTransposeKernel(&(G_lodiag[i * blockSize * blockSize]),
                                      &(G_updiag[i * blockSize * blockSize]),
                                      blockSize, cublasHandle);
            } else {

                matrixMultiplyKernel(
                    &(A_updiag[i * blockSize * blockSize]),
                    &(G_mdiag[(i + 1) * blockSize * blockSize]), Guf1,
                    blockSize, cublasHandle);
                matrixMultiplyKernel(&(G_mdiag[i * blockSize * blockSize]),
                                     Guf1, Guf, blockSize, cublasHandle);
                matrixScaleKernel<<<kernels_num_blocks, kernels_num_threads>>>(
                    Guf, -1, &(G_updiag[i * blockSize * blockSize]), blockSize);
            }
        }

        matrixMultiplyKernel(&(A_updiag[i * blockSize * blockSize]), Glf, Glf1,
                             blockSize, cublasHandle);
        matrixMultiplyKernel(&(G_mdiag[i * blockSize * blockSize]), Glf1, Glf,
                             blockSize, cublasHandle);
        matrixAddKernel<<<kernels_num_blocks, kernels_num_threads>>>(
            &(G_mdiag[i * blockSize * blockSize]), Glf,
            &(G_mdiag[i * blockSize * blockSize]), blockSize);
    }

    // Copy results back to host
    cudaMemcpy(input_G.mdiag, G_mdiag, size_mdiag, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.updiag, G_updiag, size_updiag, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.lodiag, G_lodiag, size_updiag, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(A_mat);
    cudaFree(G_mat);

    // Clean up of inverse kernel
    // free(identity_matrix);
    cudaFree(d_A);
    cudaFree(d_work);
    cudaFree(ipiv);
    cudaFree(d_identity);
    cudaFree(d_info);
    cudaFree(d_result);

    // Destroy cuBLAS handle
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);
}
