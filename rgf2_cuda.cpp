#include "rgf2_cuda.hpp"
#include "argparse.h"
#include "rgf2.hpp"
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <mpi.h>
int processRank;

extern void kernel_init(int n);
extern void matrixSubtracter(float *A, float *B, float *result, int n);
extern void matrixAdder(float *A, float *B, float *result, int n);
extern void matrixScaler(float *A, float k, float *result, int n);

float *identity_matrix;
int *d_info = nullptr;
float *d_A, *d_identity, *d_work;
int *ipiv;
float *d_result;

void printFloatArrayFromCuda(const float arr[], int size) {
    float tempResult[size];
    cudaMemcpy(tempResult, arr, sizeof(float) * size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        std::cout << tempResult[i] << " ";
    }
    std::cout << std::endl;
}

void matrixMultiplyKernel(float *A, float *B, float *result, int n,
                          cublasHandle_t cublasHandle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, B, n,
                A, n, &beta, result, n);
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

void matrixInversionKernel(float *A, float *result, int n,
                           cusolverDnHandle_t cusolverHandle) {

    // Copy the input matrix A to the device
    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_identity, identity_matrix, n * n * sizeof(float),
               cudaMemcpyHostToDevice);

    // Perform LU decomposition on the device
    cusolverDnSgetrf(cusolverHandle, n, n, d_A, n, d_work, NULL,
                     d_info); // Not using PIVOT for now

    // Solving AX = I  , where X is the result_matrix, and I is the
    // identity_matrix. Since AA^(-1) = I It saves on the result_matrix
    // (identity) the answer
    cusolverDnSgetrs(cusolverHandle, CUBLAS_OP_N, n, n, d_A, n, NULL,
                     d_identity, n, d_info); // Not using PIVOT for now

    // std::cout << "printing d_identity from CUDA after cusolverDnSgetrs: \n";
    // printFloatArrayFromCuda(d_identity, n * n);
    cudaMemcpy(result, d_identity, n * n * sizeof(float),
               cudaMemcpyDeviceToHost);
}

void matrixTransposeKernel(const float *A, float *result, int n,
                           cublasHandle_t cublasHandle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform the transposition
    cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, A, n,
                &beta, NULL, n, result, n);
}

void rgf2sided_cuda(Matrix &A, Matrix &G, bool sym_mat, bool save_off_diag) {
    int processRank, blockSize, matrixSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    int nblocks = matrixSize / blockSize; // assume divisible
    int nblocks_2 = nblocks / 2;          // assume divisible

    kernel_init(blockSize);

    if (processRank == 0) {
        rgf2sided_upperprocess_cuda(A, G, nblocks_2, sym_mat, save_off_diag);

        MPI_Recv((void *)(G.mdiag + nblocks_2 * blockSize * blockSize),
                 (nblocks_2)*blockSize * blockSize, MPI_FLOAT, 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Recv((void *)(G.updiag + nblocks_2 * blockSize * blockSize),
                 (nblocks_2 - 1) * blockSize * blockSize, MPI_FLOAT, 1, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Recv((void *)(G.lodiag + nblocks_2 * blockSize * blockSize),
                 (nblocks_2 - 1) * blockSize * blockSize, MPI_FLOAT, 1, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (processRank == 1) {
        Matrix G_dup(matrixSize, G.getMat());
        G_dup.convertDenseToBlkTridiag(blockSize);

        rgf2sided_lowerprocess_cuda(A, G_dup, nblocks - nblocks_2, sym_mat,
                                    save_off_diag);

        MPI_Send(
            (const void *)(G_dup.mdiag + nblocks_2 * blockSize * blockSize),
            (nblocks - nblocks_2) * blockSize * blockSize, MPI_FLOAT, 0, 0,
            MPI_COMM_WORLD);

        MPI_Send(
            (const void *)(G_dup.updiag + nblocks_2 * blockSize * blockSize),
            (nblocks - nblocks_2 - 1) * blockSize * blockSize, MPI_FLOAT, 0, 1,
            MPI_COMM_WORLD);

        MPI_Send(
            (const void *)(G_dup.lodiag + nblocks_2 * blockSize * blockSize),
            (nblocks - nblocks_2 - 1) * blockSize * blockSize, MPI_FLOAT, 0, 2,
            MPI_COMM_WORLD);
    }
}

void rgf2sided_upperprocess_cuda(Matrix &input_A, Matrix &input_G,
                                 int nblocks_2, bool sym_mat,
                                 bool save_off_diag) {
    int blockSize, matrixSize;
    input_A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    int nblocks = matrixSize / blockSize;
    int kernels_num_blocks = blockSize;
    int kernels_num_threads = blockSize;

    // Initialize the handle used for cuBLAS
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;

    cublasCreate(&cublasHandle);
    cusolverDnCreate(&cusolverHandle);
    // Allocate memory for Matrix specifics on the GPU
    float *A_mdiag, *G_mdiag;
    size_t size_mdiag_A = nblocks * blockSize * blockSize * sizeof(float);
    size_t size_mdiag_G =
        (nblocks_2 + 1) * blockSize * blockSize * sizeof(float);
    cudaMalloc(&A_mdiag, size_mdiag_A);
    cudaMalloc(&G_mdiag, size_mdiag_G);

    // Copy matrices from host to device
    cudaMemcpy(A_mdiag, input_A.mdiag, size_mdiag_A, cudaMemcpyHostToDevice);

    float *A_updiag, *G_updiag;
    size_t size_updiag_A =
        (nblocks - 1) * blockSize * blockSize * sizeof(float);

    size_t size_updiag_G = nblocks_2 * blockSize * blockSize * sizeof(float);
    cudaMalloc(&A_updiag, size_updiag_A);
    cudaMalloc(&G_updiag, size_updiag_G);

    // Copy matrices from host to device
    cudaMemcpy(A_updiag, input_A.updiag, size_updiag_A, cudaMemcpyHostToDevice);

    float *A_lodiag, *G_lodiag;
    cudaMalloc(&A_lodiag, size_updiag_A);
    cudaMalloc(&G_lodiag, size_updiag_G);

    // Copy matrices from host to device
    cudaMemcpy(A_lodiag, input_A.lodiag, size_updiag_A, cudaMemcpyHostToDevice);

    // Create temp result matrixes
    size_t blockSizeBytes = blockSize * blockSize * sizeof(float);
    float *temp_result_1, *temp_result_2, *temp_result_3, *temp_result_4;
    cudaMalloc(&temp_result_1, blockSizeBytes);
    cudaMalloc(&temp_result_2, blockSizeBytes);
    cudaMalloc(&temp_result_3, blockSizeBytes);
    cudaMalloc(&temp_result_4, blockSizeBytes);

    // Inverse and transpose kernel variables
    cudaMalloc(&d_info, sizeof(int));
    cudaMalloc(&d_A, blockSize * blockSize * sizeof(float));
    cudaMalloc(&d_identity, blockSize * blockSize * sizeof(float));
    cudaMalloc(&d_work, blockSize * blockSize * sizeof(float));
    cudaMalloc(&ipiv, blockSize * sizeof(int));
    cudaMalloc((void **)&d_result, blockSize * blockSize * sizeof(float));

    identity_matrix = createIdentityMatrix(blockSize);

    // Launch CUDA kernels for matrix operations

    // 0. Inverse of the first block
    matrixInversionKernel(A_mdiag, G_mdiag, blockSize, cusolverHandle);

    // 1. Forward substitution (performed left to right)
    for (int i = 1; i < nblocks_2; ++i) {
        matrixMultiplyKernel(&(A_lodiag[(i - 1) * blockSize * blockSize]),
                             &(G_mdiag[(i - 1) * blockSize * blockSize]),
                             temp_result_1, blockSize, cublasHandle);

        matrixMultiplyKernel(temp_result_1,
                             &(A_updiag[(i - 1) * blockSize * blockSize]),
                             temp_result_2, blockSize, cublasHandle);

        matrixSubtracter(&(A_mdiag[i * blockSize * blockSize]), temp_result_2,
                         temp_result_2, blockSize);

        matrixInversionKernel(temp_result_2,
                              &(G_mdiag[i * blockSize * blockSize]), blockSize,
                              cusolverHandle);
    }

    float *G_mdiag_host_send = new float[blockSize * blockSize]();
    float *G_mdiag_host_recv = new float[blockSize * blockSize]();
    cudaMemcpy(G_mdiag_host_send,
               G_mdiag + (nblocks_2 - 1) * blockSize * blockSize,
               blockSize * blockSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Communicate the left connected block and receive the right connected
    // block
    MPI_Send((const void *)G_mdiag_host_send, blockSize * blockSize, MPI_FLOAT,
             1, 0, MPI_COMM_WORLD);

    MPI_Recv((void *)G_mdiag_host_recv, blockSize * blockSize, MPI_FLOAT, 1, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    cudaMemcpy(G_mdiag + nblocks_2 * blockSize * blockSize, G_mdiag_host_recv,
               blockSize * blockSize * sizeof(float), cudaMemcpyHostToDevice);

    // Connection from both sides of the full G
    matrixMultiplyKernel(A_lodiag + (nblocks_2 - 2) * blockSize * blockSize,
                         G_mdiag + (nblocks_2 - 2) * blockSize * blockSize,
                         temp_result_1, blockSize, cublasHandle);

    matrixMultiplyKernel(temp_result_1,
                         A_updiag + (nblocks_2 - 2) * blockSize * blockSize,
                         temp_result_2, blockSize, cublasHandle);

    matrixMultiplyKernel(A_updiag + (nblocks_2 - 1) * blockSize * blockSize,
                         G_mdiag + (nblocks_2)*blockSize * blockSize,
                         temp_result_3, blockSize, cublasHandle);

    matrixMultiplyKernel(temp_result_3,
                         A_lodiag + (nblocks_2 - 1) * blockSize * blockSize,
                         temp_result_4, blockSize, cublasHandle);

    matrixSubtracter(A_mdiag + (nblocks_2 - 1) * blockSize * blockSize,
                     temp_result_2, temp_result_2, blockSize);

    matrixSubtracter(temp_result_2, temp_result_4, temp_result_2, blockSize);

    matrixInversionKernel(temp_result_2,
                          G_mdiag + (nblocks_2 - 1) * blockSize * blockSize,
                          blockSize, cusolverHandle);

    // Compute the shared off-diagonal upper block

    matrixMultiplyKernel(G_mdiag + nblocks_2 * blockSize * blockSize,
                         A_lodiag + (nblocks_2 - 1) * blockSize * blockSize,
                         temp_result_1, blockSize, cublasHandle);

    matrixMultiplyKernel(temp_result_1,
                         G_mdiag + (nblocks_2 - 1) * blockSize * blockSize,
                         temp_result_2, blockSize, cublasHandle);

    matrixScaler(temp_result_2, -1,
                 G_lodiag + (nblocks_2 - 1) * blockSize * blockSize, blockSize);

    if (sym_mat) {
        // matrix transpose
        matrixTransposeKernel(
            G_lodiag + (nblocks_2 - 1) * blockSize * blockSize,
            G_updiag + (nblocks_2 - 1) * blockSize * blockSize, blockSize,
            cublasHandle);
    } else {
        matrixMultiplyKernel(G_mdiag + (nblocks_2 - 1) * blockSize * blockSize,
                             A_updiag + (nblocks_2 - 1) * blockSize * blockSize,
                             temp_result_1, blockSize, cublasHandle);

        matrixMultiplyKernel(temp_result_1,
                             G_mdiag + (nblocks_2)*blockSize * blockSize,
                             temp_result_2, blockSize, cublasHandle);

        matrixScaler(temp_result_2, -1,
                     G_updiag + (nblocks_2 - 1) * blockSize * blockSize,
                     blockSize);
    }

    // 2. Backward substitution
    for (int i = nblocks_2 - 2; i >= 0; --i) {
        float *g_ii = G_mdiag + i * blockSize * blockSize;
        float *G_lowerfactor;
        cudaMalloc(&G_lowerfactor, blockSizeBytes);

        matrixMultiplyKernel(&(G_mdiag[(i + 1) * blockSize * blockSize]),
                             &(A_lodiag[i * blockSize * blockSize]),
                             temp_result_1, blockSize, cublasHandle);
        matrixMultiplyKernel(temp_result_1, g_ii, G_lowerfactor, blockSize,
                             cublasHandle);

        if (save_off_diag) {
            matrixScaler(G_lowerfactor, -1,
                         &(G_lodiag[i * blockSize * blockSize]), blockSize);

            if (sym_mat) {
                matrixTransposeKernel(&(G_lodiag[i * blockSize * blockSize]),
                                      &(G_updiag[i * blockSize * blockSize]),
                                      blockSize, cublasHandle);
            } else {
                matrixMultiplyKernel(g_ii, A_updiag + i * blockSize * blockSize,
                                     temp_result_1, blockSize, cublasHandle);

                matrixMultiplyKernel(temp_result_1,
                                     G_mdiag + (i + 1) * blockSize * blockSize,
                                     temp_result_2, blockSize, cublasHandle);

                matrixScaler(temp_result_2, -1,
                             &(G_updiag[i * blockSize * blockSize]), blockSize);
            }
        }

        matrixMultiplyKernel(g_ii, &(A_updiag[i * blockSize * blockSize]),
                             temp_result_1, blockSize, cublasHandle);

        matrixMultiplyKernel(temp_result_1, G_lowerfactor, temp_result_2,
                             blockSize, cublasHandle);

        matrixAdder(g_ii, temp_result_2, &(G_mdiag[i * blockSize * blockSize]),
                    blockSize);

        // Free temporary GPU memory
        cudaFree(G_lowerfactor);
    }

    // Copy results back to host
    cudaMemcpy(input_G.mdiag, G_mdiag,
               (nblocks_2 + 1) * blockSize * blockSize * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.updiag, G_updiag,
               nblocks_2 * blockSize * blockSize * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.lodiag, G_lodiag,
               nblocks_2 * blockSize * blockSize * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Free GPU memory
    // cudaFree(A);
    // cudaFree(G);
    cudaFree(A_mdiag);
    cudaFree(G_mdiag);
    cudaFree(A_updiag);
    cudaFree(G_updiag);
    cudaFree(A_lodiag);
    cudaFree(G_lodiag);
    cudaFree(temp_result_1);
    cudaFree(temp_result_2);
    cudaFree(temp_result_3);
    cudaFree(temp_result_4);

    // Clean up of inverse kernel
    free(identity_matrix);
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

void rgf2sided_lowerprocess_cuda(Matrix &input_A, Matrix &input_G,
                                 int nblocks_2, bool sym_mat,
                                 bool save_off_diag) {

    int blockSize, matrixSize;
    input_A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    int nblocks = matrixSize / blockSize;
    int kernels_num_blocks = blockSize;
    int kernels_num_threads = blockSize;

    // Initialize the handle used for cuBLAS
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;

    cublasCreate(&cublasHandle);
    cusolverDnCreate(&cusolverHandle);

    // Allocate memory for Matrix specifics on the GPU
    float *A_mdiag, *G_mdiag;
    size_t size_mdiag_A =
        (nblocks - nblocks_2) * blockSize * blockSize * sizeof(float);
    size_t size_mdiag_G =
        (nblocks_2 + 1) * blockSize * blockSize * sizeof(float);
    cudaMalloc(&A_mdiag, size_mdiag_A);
    cudaMalloc(&G_mdiag, size_mdiag_G);

    // Copy matrices from host to device
    cudaMemcpy(A_mdiag, input_A.mdiag + nblocks_2 * blockSize * blockSize,
               size_mdiag_A, cudaMemcpyHostToDevice);

    float *A_updiag, *G_updiag;
    size_t size_updiag_A =
        (nblocks - nblocks_2 - 1) * blockSize * blockSize * sizeof(float);
    size_t size_updiag_G = nblocks_2 * blockSize * blockSize * sizeof(float);
    cudaMalloc(&A_updiag, size_updiag_A);
    cudaMalloc(&G_updiag, size_updiag_G);

    // Copy matrices from host to device
    cudaMemcpy(A_updiag,
               input_A.updiag + (nblocks_2 - 1) * blockSize * blockSize,
               size_updiag_A, cudaMemcpyHostToDevice);

    float *A_lodiag, *G_lodiag;
    cudaMalloc(&A_lodiag, size_updiag_A);
    cudaMalloc(&G_lodiag, size_updiag_G);

    // Copy matrices from host to device
    cudaMemcpy(A_lodiag,
               input_A.lodiag + (nblocks_2 - 1) * blockSize * blockSize,
               size_updiag_A, cudaMemcpyHostToDevice);

    // Create temp result matrixes
    size_t blockSizeBytes = blockSize * blockSize * sizeof(float);
    float *temp_result_1, *temp_result_2, *temp_result_3, *temp_result_4;
    cudaMalloc(&temp_result_1, blockSizeBytes);
    cudaMalloc(&temp_result_2, blockSizeBytes);
    cudaMalloc(&temp_result_3, blockSizeBytes);
    cudaMalloc(&temp_result_4, blockSizeBytes);

    // Inverse and transpose kernel variables
    cudaMalloc(&d_info, sizeof(int));
    cudaMalloc(&d_A, blockSize * blockSize * sizeof(float));
    cudaMalloc(&d_identity, blockSize * blockSize * sizeof(float));
    cudaMalloc(&d_work, blockSize * blockSize * sizeof(float));
    cudaMalloc(&ipiv, blockSize * sizeof(int));
    cudaMalloc((void **)&d_result, blockSize * blockSize * sizeof(float));

    identity_matrix = createIdentityMatrix(blockSize);

    // Launch CUDA kernels for matrix operations

    // 0. Inverse of the first block
    matrixInversionKernel(A_mdiag + (nblocks_2 - 1) * blockSize * blockSize,
                          G_mdiag + nblocks_2 * blockSize * blockSize,
                          blockSize, cusolverHandle);

    // 1. Forward substitution (performed left to right)
    for (int i = nblocks_2 - 1; i >= 1; i -= 1) {
        matrixMultiplyKernel(A_updiag + i * blockSize * blockSize,
                             G_mdiag + (i + 1) * blockSize * blockSize,
                             temp_result_1, blockSize, cublasHandle);

        matrixMultiplyKernel(temp_result_1,
                             A_lodiag + i * blockSize * blockSize,
                             temp_result_2, blockSize, cublasHandle);

        matrixSubtracter(&(A_mdiag[(i - 1) * blockSize * blockSize]),
                         temp_result_2, temp_result_2, blockSize);

        matrixInversionKernel(temp_result_2,
                              &(G_mdiag[i * blockSize * blockSize]), blockSize,
                              cusolverHandle);
    }

    // Communicate the right connected block and receive the right connected
    // block

    float *G_mdiag_host_send = new float[blockSize * blockSize]();
    float *G_mdiag_host_recv = new float[blockSize * blockSize]();

    MPI_Recv((void *)(G_mdiag_host_recv), blockSize * blockSize, MPI_FLOAT, 0,
             0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    cudaMemcpy(G_mdiag, G_mdiag_host_recv,
               blockSize * blockSize * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(G_mdiag_host_send, G_mdiag + 1 * blockSize * blockSize,
               blockSize * blockSize * sizeof(float), cudaMemcpyDeviceToHost);

    MPI_Send((const void *)G_mdiag_host_send, blockSize * blockSize, MPI_FLOAT,
             0, 0, MPI_COMM_WORLD);

    // Connection from both sides of the full G
    matrixMultiplyKernel(A_lodiag, G_mdiag, temp_result_1, blockSize,
                         cublasHandle);

    matrixMultiplyKernel(temp_result_1, A_updiag, temp_result_2, blockSize,
                         cublasHandle);

    matrixMultiplyKernel(A_updiag + (1) * blockSize * blockSize,
                         G_mdiag + (2) * blockSize * blockSize, temp_result_3,
                         blockSize, cublasHandle);

    matrixMultiplyKernel(temp_result_3, A_lodiag + (1) * blockSize * blockSize,
                         temp_result_4, blockSize, cublasHandle);

    matrixSubtracter(A_mdiag, temp_result_2, temp_result_2, blockSize);

    matrixSubtracter(temp_result_2, temp_result_4, temp_result_2, blockSize);

    matrixInversionKernel(temp_result_2, G_mdiag + (1) * blockSize * blockSize,
                          blockSize, cusolverHandle);

    // Compute the shared off-diagonal upper block

    matrixMultiplyKernel(G_mdiag + (1) * blockSize * blockSize, A_lodiag,
                         temp_result_1, blockSize, cublasHandle);

    matrixMultiplyKernel(temp_result_1, G_mdiag, G_lodiag, blockSize,
                         cublasHandle);

    if (sym_mat) {
        // matrix transpose
        matrixTransposeKernel(G_lodiag, G_updiag, blockSize, cublasHandle);
    } else {
        matrixMultiplyKernel(G_mdiag, A_updiag, temp_result_1, blockSize,
                             cublasHandle);

        matrixMultiplyKernel(temp_result_1,
                             G_mdiag + (1) * blockSize * blockSize, G_updiag,
                             blockSize, cublasHandle);
    }

    // 2. Backward substitution
    for (int i = 1; i < nblocks_2; ++i) {
        float *g_ii = G_mdiag + (i + 1) * blockSize * blockSize;
        float *G_lowerfactor;
        cudaMalloc(&G_lowerfactor, blockSizeBytes);

        matrixMultiplyKernel(g_ii, &(A_lodiag[i * blockSize * blockSize]),
                             temp_result_1, blockSize, cublasHandle);

        matrixMultiplyKernel(temp_result_1, G_mdiag + i * blockSize * blockSize,
                             G_lowerfactor, blockSize, cublasHandle);

        if (save_off_diag) {
            matrixScaler(G_lowerfactor, -1,
                         &(G_lodiag[i * blockSize * blockSize]), blockSize);

            if (sym_mat) {
                matrixTransposeKernel(&(G_lodiag[i * blockSize * blockSize]),
                                      &(G_updiag[i * blockSize * blockSize]),
                                      blockSize, cublasHandle);
            } else {
                matrixMultiplyKernel(G_mdiag + i * blockSize * blockSize,
                                     A_updiag + i * blockSize * blockSize,
                                     temp_result_1, blockSize, cublasHandle);

                matrixMultiplyKernel(temp_result_1, g_ii, temp_result_2,
                                     blockSize, cublasHandle);

                matrixScaler(temp_result_2, -1,
                             &(G_updiag[i * blockSize * blockSize]), blockSize);
            }
        }

        matrixMultiplyKernel(G_lowerfactor,
                             &(A_updiag[i * blockSize * blockSize]),
                             temp_result_1, blockSize, cublasHandle);

        matrixMultiplyKernel(temp_result_1, g_ii, temp_result_2, blockSize,
                             cublasHandle);

        matrixAdder(g_ii, temp_result_2,
                    &(G_mdiag[(i + 1) * blockSize * blockSize]), blockSize);

        // Free temporary GPU memory
        cudaFree(G_lowerfactor);
    }

    // Copy results back to host
    cudaMemcpy(input_G.mdiag + nblocks_2 * blockSize * blockSize,
               G_mdiag + blockSize * blockSize,
               (nblocks_2)*blockSize * blockSize * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.updiag + (nblocks_2 - 1) * blockSize * blockSize,
               G_updiag, nblocks_2 * blockSize * blockSize * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.lodiag + (nblocks_2 - 1) * blockSize * blockSize,
               G_lodiag, nblocks_2 * blockSize * blockSize * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Free GPU memory
    // cudaFree(A);
    // cudaFree(G);
    cudaFree(A_mdiag);
    cudaFree(G_mdiag);
    cudaFree(A_updiag);
    cudaFree(G_updiag);
    cudaFree(A_lodiag);
    cudaFree(G_lodiag);
    cudaFree(temp_result_1);
    cudaFree(temp_result_2);
    cudaFree(temp_result_3);
    cudaFree(temp_result_4);

    // Clean up of inverse kernel
    free(identity_matrix);
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

typedef struct {
    int matrixSize;
    int blockSize;
    int numRuns;
    bool isSymmetric;
    bool saveOffDiag;
    char *inputPath;
} Config;

void InitOptions(Config *config) {
    config->blockSize = 2;
    config->matrixSize = 0;
    config->numRuns = 10;
    config->isSymmetric = false;
    config->saveOffDiag = true;
    config->inputPath = NULL;
}

int parse(Config *config, int argc, const char **argv) {
    static const char *const usages[] = {
        NULL,
    };
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_INTEGER('m', "matrixSize", &config->matrixSize, "matrix size", NULL,
                    0, 0),
        OPT_INTEGER('b', "blockSize", &config->blockSize, "block size", NULL, 0,
                    0),
        OPT_INTEGER('n', "numRuns", &config->numRuns, "number of runs", NULL, 0,
                    0),
        OPT_INTEGER('s', "isSymmetric", &config->isSymmetric, "is symmetric",
                    NULL, 0, 0),
        OPT_INTEGER('o', "saveOffDiag", &config->saveOffDiag, "save off diag",
                    NULL, 0, 0),
        OPT_STRING('f', "inputPath", &config->inputPath, "input path", NULL, 0,
                   0),
        OPT_END(),
    };

    struct argparse argparse;
    argparse_init(&argparse, options, usages, 0);
    argparse_describe(&argparse, "DPHPC TEAM", NULL);
    argc = argparse_parse(&argparse, argc, argv);

    return 0;
}

// // TEMP main to test stuff out
// int main(int argc, const char *argv[]) {
//     const char *bin_name = argv[0];
//     Config config;
//     InitOptions(&config);
//     parse(&config, argc, argv);

//     MPI_Init(&argc, (char ***)(&argv));
//     MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

//     if (config.inputPath != NULL) {
//         // read matrix from file
//     } else if (config.matrixSize != 0) {
//         // generate matrix
//         int MATRIX_SIZE = config.matrixSize;
//         int BLOCK_SIZE = config.blockSize;
//         assert(MATRIX_SIZE % BLOCK_SIZE == 0);
//         int NUM_RUNS = config.numRuns;
//         bool IS_SYMMETRIC = config.isSymmetric;
//         bool SAVE_OFF_DIAG = config.saveOffDiag;

//         Matrix inputMatrix =
//             generateBandedDiagonalMatrix(MATRIX_SIZE, 2, true, 0);
//         // Matrix inputMatrix = generateFixedMatrixOfSize4();
//         // inputMatrix.convertDenseToBlkTridiag(BLOCK_SIZE);

//         // if (processRank == 0) {
//         //     inputMatrix.printB();
//         // }

//         Matrix tempResult(
//             MATRIX_SIZE); // zero initialization, same shape as inputMatrix
//         tempResult.convertDenseToBlkTridiag(
//             BLOCK_SIZE); // G has same blockSize as inputMatrix
//         rgf2sided_cuda(inputMatrix, tempResult, IS_SYMMETRIC, SAVE_OFF_DIAG);

//         if (processRank == 0) {
//             std::cout << "\n\nCUDA RESULT\n\n";
//             tempResult.printB();
//         }

//         // Check against the already implemented RGF2 on C++
//         Matrix tempResult_cpp(
//             MATRIX_SIZE); // zero initialization, same shape as inputMatrix
//         tempResult_cpp.convertDenseToBlkTridiag(
//             BLOCK_SIZE); // G has same blockSize as inputMatrix

//         rgf2sided(inputMatrix, tempResult_cpp, false, true);

//         if (processRank == 0) {
//             std::cout << "\n\nC++ RESULT \n\n";
//             tempResult_cpp.printB();
//         }
//     }
//     MPI_Finalize();
// }