#include "rgf2.hpp"
#include "rgf2_cuda.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

void printFloatArray(const float arr[], int size) {
    // std::cout << "Array of floats: \n";
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

void printFloatArrayFromCuda(const float arr[], int size) {
    float tempResult[size];
    cudaMemcpy(tempResult, arr, sizeof(float) * size, cudaMemcpyDeviceToHost);
    // std::cout << "Array of floats from GPU: \n";
    for (int i = 0; i < size; ++i) {
        std::cout << tempResult[i] << " ";
    }
    std::cout << std::endl;
}

void matrixMultiplyKernel(float *A, float *B, float *result, int n,
                          cublasHandle_t cublasHandle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n,
                B, n, &beta, result, n);
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
    float *identity_matrix = createIdentityMatrix(n);
    int *d_info = nullptr; /* error info */
    cudaMalloc(&d_info, sizeof(int));

    // Create a temporary matrix on the device
    float *d_A, *d_identity, *d_work;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_identity, n * n * sizeof(float));
    cudaMalloc(&d_work, n * n * sizeof(float));
    int *ipiv;
    cudaMalloc(&ipiv, n * sizeof(int));

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

    // Clean up
    free(identity_matrix);
    cudaFree(d_A);
    cudaFree(d_work);
    cudaFree(ipiv);
    cudaFree(d_identity);
    cudaFree(d_info);
}

void matrixTransposeKernel(const float *A, float *result, int n,
                           cublasHandle_t cublasHandle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Allocate device memory for the input and output matrices
    float *d_A, *d_result;
    cudaMalloc((void **)&d_A, n * n * sizeof(float));
    cudaMalloc((void **)&d_result, n * n * sizeof(float));

    // Copy the input matrix A to device memory
    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Perform the transposition
    cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, d_A, n,
                &beta, NULL, n, d_result, n);

    // Copy the transposed matrix back to the host memory
    cudaMemcpy(result, d_result, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_result);
}

void rgf2sided_cuda(Matrix &A, Matrix &G,
                    bool sym_mat,
                    bool save_off_diag) {
    int processRank, blockSize, matrixSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    int nblocks = matrixSize / blockSize; // assume divisible
    int nblocks_2 = nblocks / 2;          // assume divisible

    if (processRank == 0) {
        //TODO make the CUDA implementation with rgf2sided_upperprocess_cuda
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
        
        //TODO make the CUDA implementation with rgf2sided_lowerprocess_cuda
        rgf2sided_lowerprocess_cuda (A, G_dup, nblocks - nblocks_2, sym_mat,
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

void rgf2sided_upperprocess_cuda(Matrix &input_A, Matrix &input_G, int nblocks_2,
                                 bool sym_mat,
                                 bool save_off_diag) {
    int blockSize, matrixSize;
    input_A.getBlockSizeAndMatrixSize(blockSize, matrixSize);

    // Initialize the handle used for cuBLAS
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;

    cublasCreate(&cublasHandle);
    cusolverDnCreate(&cusolverHandle);

    // Allocate memory for matrices on the GPU
    float *A, *G;
    size_t size = matrixSize * matrixSize * sizeof(float);
    int matrix_array_size = matrixSize * matrixSize;
    cudaMalloc(&A, size);
    cudaMalloc(&G, size);

    // Copy matrices from host to device
    cudaMemcpy(A, input_A.getMat(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(G, input_G.getMat(), size, cudaMemcpyHostToDevice);

    // Allocate memory for Matrix specifics on the GPU
    float *A_mdiag, *G_mdiag;
    size_t size_mdiag = nblocks_2 * blockSize * blockSize * sizeof(float);
    cudaMalloc(&A_mdiag, size_mdiag);
    cudaMalloc(&G_mdiag, size_mdiag);

    // Copy matrices from host to device
    cudaMemcpy(A_mdiag, input_A.mdiag, size_mdiag, cudaMemcpyHostToDevice);
    cudaMemcpy(G_mdiag, input_G.mdiag, size_mdiag, cudaMemcpyHostToDevice);

    float *A_updiag, *G_updiag;
    size_t size_updiag = (nblocks_2 - 1) * blockSize * blockSize * sizeof(float);
    cudaMalloc(&A_updiag, size_updiag);
    cudaMalloc(&G_updiag, size_updiag);

    // Copy matrices from host to device
    cudaMemcpy(A_updiag, input_A.updiag, size_updiag, cudaMemcpyHostToDevice);
    cudaMemcpy(G_updiag, input_G.updiag, size_updiag, cudaMemcpyHostToDevice);

    float *A_lodiag, *G_lodiag;
    cudaMalloc(&A_lodiag, size_updiag);
    cudaMalloc(&G_lodiag, size_updiag);

    // Copy matrices from host to device
    cudaMemcpy(A_lodiag, input_A.lodiag, size_updiag, cudaMemcpyHostToDevice);
    cudaMemcpy(G_lodiag, input_G.lodiag, size_updiag, cudaMemcpyHostToDevice);

    // Create temp result matrixes
    size_t blockSizeBytes = blockSize * blockSize * sizeof(float);
    float *temp_result_1, *temp_result_2, *temp_result_3, *temp_result_4;
    cudaMalloc(&temp_result_1, blockSizeBytes);
    cudaMalloc(&temp_result_2, blockSizeBytes);
    cudaMalloc(&temp_result_3, blockSizeBytes);
    cudaMalloc(&temp_result_4, blockSizeBytes);

    // Launch CUDA kernels for matrix operations

    // 0. Inverse of the first block
    matrixInversionKernel(A_mdiag, G_mdiag, blockSize, cusolverHandle);

    int kernels_num_blocks = nblocks_2;
    int kernels_num_threads = nblocks_2;
    // 1. Forward substitution (performed left to right)
    for (int i = 1; i < nblocks_2; ++i) {
        // TODO, check how to parallelize, since u need the previous G
        matrixMultiplyKernel(&(A_lodiag[(i - 1) * blockSize * blockSize]),
                             &(G_mdiag[(i - 1) * blockSize * blockSize]), temp_result_1,
                             blockSize, cublasHandle);
        matrixMultiplyKernel(temp_result_1, &(A_updiag[(i - 1) * blockSize * blockSize]),
                             temp_result_2, blockSize, cublasHandle);
        matrixSubtractKernel<<<kernels_num_blocks, kernels_num_threads>>>(
            &(A_mdiag[i * blockSize * blockSize]), temp_result_2, temp_result_2, blockSize);
        matrixInversionKernel(temp_result_2, &(G_mdiag[i * blockSize * blockSize]),
                              blockSize, cusolverHandle);
    }

    // Communicate the left connected block and receive the right connected
    // block
    MPI_Send((const void *)(G_mdiag +
                            (nblocks_2 - 1) * blockSize * blockSize),
             blockSize * blockSize, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(
        (void *)(G_mdiag + nblocks_2 * blockSize * blockSize),
        blockSize * blockSize, MPI_FLOAT, 1, 0, MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);

    // Connection from both sides of the full G
    matrixMultiplyKernel(A_lodiag + (nblocks_2 - 2) * blockSize * blockSize,
                            G_mdiag + (nblocks_2 - 2) * blockSize * blockSize, 
                            temp_result_1,
                            blockSize, cublasHandle);
    matrixMultiplyKernel(temp_result_1,
                            A_updiag + (nblocks_2 - 2) * blockSize * blockSize, 
                            temp_result_2,
                            blockSize, cublasHandle);
    matrixMultiplyKernel(A_updiag + (nblocks_2 - 1) * blockSize * blockSize,
                            G_mdiag  + (nblocks_2)*blockSize * blockSize, 
                            temp_result_3,
                            blockSize, cublasHandle);
    matrixMultiplyKernel(temp_result_3,
                            A_lodiag  + (nblocks_2 - 1)*blockSize * blockSize, 
                            temp_result_4,
                            blockSize, cublasHandle);
                            
    matrixSubtractKernel<<<kernels_num_blocks, kernels_num_threads>>>(
            A_mdiag + (nblocks_2 - 1) * blockSize * blockSize, temp_result_2, temp_result_2, blockSize);
    matrixSubtractKernel<<<kernels_num_blocks, kernels_num_threads>>>(
            temp_result_2, temp_result_4, temp_result_2, blockSize);
    matrixInversionKernel(temp_result_2, G_mdiag + (nblocks_2 - 1) * blockSize * blockSize,
                          blockSize, cusolverHandle);

    // Compute the shared off-diagonal upper block
    
    matrixMultiplyKernel(G_mdiag + nblocks_2 * blockSize * blockSize,
                            A_lodiag + (nblocks_2 - 1) * blockSize * blockSize,
                            temp_result_1,
                            blockSize, cublasHandle);
    matrixMultiplyKernel(temp_result_1,
                            G_mdiag + (nblocks_2 - 1) * blockSize * blockSize,
                            temp_result_2,
                            blockSize, cublasHandle);
    matrixScaleKernel<<<kernels_num_blocks, kernels_num_threads>>>(
            temp_result_2, -1, G_lodiag + (nblocks_2 - 1) * blockSize * blockSize, blockSize);
    if (sym_mat) {
        // matrix transpose
        
        matrixTransposeKernel(G_lodiag + (nblocks_2 - 1) * blockSize * blockSize,
                                G_updiag + (nblocks_2 - 1) * blockSize * blockSize,
                                blockSize, cublasHandle);
    } else {
    
        matrixMultiplyKernel(G_mdiag + (nblocks_2 - 1) * blockSize * blockSize,
                            A_updiag + (nblocks_2 - 1) * blockSize * blockSize,
                            temp_result_1,
                            blockSize, cublasHandle);
        matrixMultiplyKernel(temp_result_1,
                            G_mdiag + (nblocks_2) * blockSize * blockSize,
                            temp_result_2,
                            blockSize, cublasHandle);
        matrixScaleKernel<<<kernels_num_blocks, kernels_num_threads>>>(
            temp_result_2, -1, G_updiag + (nblocks_2 - 1) * blockSize * blockSize, blockSize);
    
    }

    // 2. Backward substitution

    for (int i = nblocks_2 - 2; i >= 0; --i) {
        float *g_ii = G_mdiag + i * blockSize * blockSize;
        float *G_lowerfactor;
        cudaMalloc(&G_lowerfactor, blockSizeBytes);

        matrixMultiplyKernel(&(G_mdiag[(i + 1) * blockSize * blockSize]),
                             &(A_lodiag[i * blockSize * blockSize]), temp_result_1,
                             blockSize, cublasHandle);
        matrixMultiplyKernel(temp_result_1, g_ii, G_lowerfactor,
                             blockSize, cublasHandle);

        if (save_off_diag) {
            matrixScaleKernel<<<kernels_num_blocks, kernels_num_threads>>>(
                G_lowerfactor, -1, &(G_lodiag[i * blockSize * blockSize]), blockSize);

            if (sym_mat) {
                matrixTransposeKernel(&(G_lodiag[i * blockSize * blockSize]),
                                      &(G_updiag[i * blockSize * blockSize]),
                                      blockSize, cublasHandle);
            } else {
                matrixMultiplyKernel(g_ii,
                                A_updiag + i * blockSize * blockSize,
                                temp_result_1,
                                blockSize, cublasHandle);
                matrixMultiplyKernel(temp_result_1,
                                G_mdiag + (i + 1) * blockSize * blockSize,
                                temp_result_2,
                                blockSize, cublasHandle);
                matrixScaleKernel<<<kernels_num_blocks, kernels_num_threads>>>(
                    temp_result_2, -1, &(G_updiag[i * blockSize * blockSize]), blockSize);
            }
        }

        matrixMultiplyKernel(g_ii,
                            &(A_updiag[i * blockSize * blockSize]), 
                            temp_result_1,
                            blockSize, cublasHandle);
        matrixMultiplyKernel(temp_result_1,
                            G_lowerfactor,
                            temp_result_2,
                            blockSize, cublasHandle);
        matrixAddKernel<<<kernels_num_blocks, kernels_num_threads>>>(
            g_ii, temp_result_2,
            &(G_mdiag[i * blockSize * blockSize]), blockSize);

            
        // Free temporary GPU memory
        cudaFree(G_lowerfactor);
    }

    //printFloatArrayFromCuda(G, matrix_array_size);

    // Copy results back to host
    cudaMemcpy(input_G.mdiag, G_mdiag, nblocks_2 * blockSize * blockSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.updiag, G_updiag, nblocks_2 * blockSize * blockSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.lodiag, G_lodiag, nblocks_2 * blockSize * blockSize * sizeof(float), cudaMemcpyDeviceToHost);

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
    cusolverDnDestroy(cusolverHandle);
}

void rgf2sided_lowerprocess_cuda(Matrix &input_A, Matrix &input_G, int nblocks_2,
                                 bool sym_mat,
                                 bool save_off_diag) {
    int blockSize, matrixSize;
    input_A.getBlockSizeAndMatrixSize(blockSize, matrixSize);

    // Initialize the handle used for cuBLAS
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;

    cublasCreate(&cublasHandle);
    cusolverDnCreate(&cusolverHandle);

    // Allocate memory for matrices on the GPU
    float *A, *G;
    size_t size = matrixSize * matrixSize * sizeof(float);
    int matrix_array_size = matrixSize * matrixSize;
    cudaMalloc(&A, size);
    cudaMalloc(&G, size);

    // Copy matrices from host to device
    cudaMemcpy(A, input_A.getMat(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(G, input_G.getMat(), size, cudaMemcpyHostToDevice);

    // Allocate memory for Matrix specifics on the GPU
    float *A_mdiag, *G_mdiag;
    size_t size_mdiag = nblocks_2 * blockSize * blockSize * sizeof(float);
    cudaMalloc(&A_mdiag, size_mdiag);
    cudaMalloc(&G_mdiag, size_mdiag);

    // Copy matrices from host to device
    cudaMemcpy(A_mdiag, input_A.mdiag + nblocks_2 * blockSize *
                      blockSize, size_mdiag, cudaMemcpyHostToDevice);
    cudaMemcpy(G_mdiag, input_G.mdiag + nblocks_2 * blockSize *
                      blockSize, size_mdiag, cudaMemcpyHostToDevice);

    float *A_updiag, *G_updiag;
    size_t size_updiag = (nblocks_2 - 1) * blockSize * blockSize * sizeof(float);
    cudaMalloc(&A_updiag, size_updiag);
    cudaMalloc(&G_updiag, size_updiag);

    // Copy matrices from host to device
    cudaMemcpy(A_updiag, input_A.updiag + (nblocks_2 - 1) * blockSize * blockSize, size_updiag, cudaMemcpyHostToDevice);
    cudaMemcpy(G_updiag, input_G.updiag + (nblocks_2 - 1) * blockSize * blockSize, size_updiag, cudaMemcpyHostToDevice);

    float *A_lodiag, *G_lodiag;
    cudaMalloc(&A_lodiag, size_updiag);
    cudaMalloc(&G_lodiag, size_updiag);

    // Copy matrices from host to device
    cudaMemcpy(A_lodiag, input_A.lodiag + (nblocks_2 - 1) * blockSize * blockSize, size_updiag, cudaMemcpyHostToDevice);
    cudaMemcpy(G_lodiag, input_G.lodiag + (nblocks_2 - 1) * blockSize * blockSize, size_updiag, cudaMemcpyHostToDevice);

    // Create temp result matrixes
    size_t blockSizeBytes = blockSize * blockSize * sizeof(float);
    float *temp_result_1, *temp_result_2, *temp_result_3, *temp_result_4;
    cudaMalloc(&temp_result_1, blockSizeBytes);
    cudaMalloc(&temp_result_2, blockSizeBytes);
    cudaMalloc(&temp_result_3, blockSizeBytes);
    cudaMalloc(&temp_result_4, blockSizeBytes);

    // Launch CUDA kernels for matrix operations

    // 0. Inverse of the first block
    matrixInversionKernel(A_mdiag + (nblocks_2 - 1) * blockSize * blockSize, 
                        G_mdiag + nblocks_2 * blockSize * blockSize,
                        blockSize, cusolverHandle);

    int kernels_num_blocks = nblocks_2;
    int kernels_num_threads = nblocks_2;
    // 1. Forward substitution (performed left to right)
    for (int i = nblocks_2 - 1; i >= 1; i -= 1) {
        // TODO, check how to parallelize, since u need the previous G
        matrixMultiplyKernel(A_updiag + i * blockSize * blockSize,
                            G_mdiag + (i + 1) * blockSize * blockSize,
                            temp_result_1,
                            blockSize, cublasHandle);
        matrixMultiplyKernel(temp_result_1,
                            A_lodiag + i * blockSize * blockSize,
                            temp_result_2,
                            blockSize, cublasHandle);
        matrixSubtractKernel<<<kernels_num_blocks, kernels_num_threads>>>(
            &(A_mdiag[(i - 1) * blockSize * blockSize]), temp_result_2, temp_result_2, blockSize);
        matrixInversionKernel(temp_result_2, &(G_mdiag[i * blockSize * blockSize]),
                              blockSize, cusolverHandle);

    }

    // Communicate the right connected block and receive the right connected
    // block

    MPI_Recv((void *)(G_mdiag), blockSize * blockSize, MPI_FLOAT,
             0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send((const void *)(G_mdiag + 1 * blockSize * blockSize),
             blockSize * blockSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

    // Connection from both sides of the full G
    matrixMultiplyKernel(A_lodiag,
                            G_mdiag, 
                            temp_result_1,
                            blockSize, cublasHandle);
    matrixMultiplyKernel(temp_result_1,
                            A_updiag, 
                            temp_result_2,
                            blockSize, cublasHandle);
    matrixMultiplyKernel(A_updiag + (1) * blockSize * blockSize,
                            G_mdiag + (2) * blockSize * blockSize, 
                            temp_result_3,
                            blockSize, cublasHandle);
    matrixMultiplyKernel(temp_result_3,
                            A_lodiag  + (1)*blockSize * blockSize, 
                            temp_result_4,
                            blockSize, cublasHandle);
                            
    matrixSubtractKernel<<<kernels_num_blocks, kernels_num_threads>>>(
            A_mdiag, temp_result_2, temp_result_2, blockSize);
    matrixSubtractKernel<<<kernels_num_blocks, kernels_num_threads>>>(
            temp_result_2, temp_result_4, temp_result_2, blockSize);
    matrixInversionKernel(temp_result_2, G_mdiag + (1) * blockSize * blockSize,
                          blockSize, cusolverHandle);

    // Compute the shared off-diagonal upper block
    
    matrixMultiplyKernel(G_mdiag + (1) * blockSize * blockSize,
                            A_lodiag,
                            temp_result_1,
                            blockSize, cublasHandle);
    matrixMultiplyKernel(temp_result_1,
                            G_mdiag,
                            G_lodiag,
                            blockSize, cublasHandle);
    matrixScaleKernel<<<kernels_num_blocks, kernels_num_threads>>>(
            temp_result_2, -1, G_lodiag + (nblocks_2 - 1) * blockSize * blockSize, blockSize);
    if (sym_mat) {
        // matrix transpose
        matrixTransposeKernel(G_lodiag,
                                G_updiag,
                                blockSize, cublasHandle);
    } else {
    
        matrixMultiplyKernel(G_mdiag,
                            A_updiag,
                            temp_result_1,
                            blockSize, cublasHandle);
        matrixMultiplyKernel(temp_result_1,
                            G_mdiag + (1) * blockSize * blockSize,
                            G_updiag,
                            blockSize, cublasHandle);
        matrixScaleKernel<<<kernels_num_blocks, kernels_num_threads>>>(
            temp_result_2, -1, G_updiag + (nblocks_2 - 1) * blockSize * blockSize, blockSize);
    
    }

    // 2. Backward substitution
    for (int i = 1; i < nblocks_2; ++i) {
        float *g_ii = G_mdiag + (i + 1) * blockSize * blockSize;
        float *G_lowerfactor;
        cudaMalloc(&G_lowerfactor, blockSizeBytes);

        matrixMultiplyKernel(g_ii,
                             &(A_lodiag[i * blockSize * blockSize]),
                             temp_result_1,
                             blockSize, cublasHandle);
        matrixMultiplyKernel(temp_result_1,
                             G_lodiag + i * blockSize * blockSize, 
                             G_lowerfactor,
                             blockSize, cublasHandle);

        if (save_off_diag) {
            matrixScaleKernel<<<kernels_num_blocks, kernels_num_threads>>>(
                G_lowerfactor, -1, &(G_lodiag[i * blockSize * blockSize]), blockSize);

            if (sym_mat) {
                matrixTransposeKernel(&(G_lodiag[i * blockSize * blockSize]),
                                      &(G_updiag[i * blockSize * blockSize]),
                                      blockSize, cublasHandle);
            } else {
                matrixMultiplyKernel(G_mdiag + i * blockSize * blockSize,
                                A_updiag + i * blockSize * blockSize,
                                temp_result_1,
                                blockSize, cublasHandle);
                matrixMultiplyKernel(temp_result_1,
                                g_ii,
                                temp_result_2,
                                blockSize, cublasHandle);
                matrixScaleKernel<<<kernels_num_blocks, kernels_num_threads>>>(
                    temp_result_2, -1, &(G_updiag[i * blockSize * blockSize]), blockSize);
            }
        }

        matrixMultiplyKernel(G_lowerfactor,
                            &(A_updiag[i * blockSize * blockSize]), 
                            temp_result_1,
                            blockSize, cublasHandle);
        matrixMultiplyKernel(temp_result_1,
                            g_ii,
                            temp_result_2,
                            blockSize, cublasHandle);
        matrixAddKernel<<<kernels_num_blocks, kernels_num_threads>>>(
            g_ii, temp_result_2,
            &(G_mdiag[(i + 1) * blockSize * blockSize]), blockSize);

            
        // Free temporary GPU memory
        cudaFree(G_lowerfactor);
    }

    //printFloatArrayFromCuda(G, matrix_array_size);

    // Copy results back to host
    cudaMemcpy(input_G.mdiag + nblocks_2 * blockSize * blockSize,
                G_mdiag + blockSize * blockSize,
                nblocks_2 * blockSize * blockSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.updiag + (nblocks_2 - 1) * blockSize * blockSize,
                G_updiag, 
                nblocks_2 * blockSize * blockSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.lodiag + (nblocks_2 - 1) * blockSize * blockSize,
                G_lodiag,
                nblocks_2 * blockSize * blockSize * sizeof(float), cudaMemcpyDeviceToHost);

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
    cusolverDnDestroy(cusolverHandle);
}

// TEMP main to test stuff out
int main(int argc, const char *argv[]) {
    int MATRIX_SIZE = 4;
    int BLOCK_SIZE = 2;
    bool IS_SYMMETRIC = false;
    bool SAVE_OFF_DIAG = true;

    int processRank;
    MPI_Init(&argc, (char ***)(&argv));
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    // Matrix inputMatrix = generateBandedDiagonalMatrix(MATRIX_SIZE, 2, true,
    // 0);
    Matrix inputMatrix = generateFixedMatrixOfSize4();
    inputMatrix.convertDenseToBlkTridiag(BLOCK_SIZE);

    if (processRank == 0) {
        inputMatrix.printB();
    }

    Matrix tempResult(
        MATRIX_SIZE); // zero initialization, same shape as inputMatrix
    tempResult.convertDenseToBlkTridiag(
        BLOCK_SIZE); // G has same blockSize as inputMatrix
    rgf2sided_cuda(inputMatrix, tempResult, IS_SYMMETRIC, SAVE_OFF_DIAG);

    if (processRank == 0) {
        tempResult.printB();
        std::cout << "########################################## \n";
    }

    inputMatrix.printB();
    // Check against the already implemented RGF1 on C++
    Matrix tempResult_cpp(
        MATRIX_SIZE); // zero initialization, same shape as inputMatrix
    tempResult_cpp.convertDenseToBlkTridiag(
        BLOCK_SIZE); // G has same blockSize as inputMatrix

    rgf2sided(inputMatrix, tempResult_cpp, false, true);

    if (processRank == 0) {
        tempResult_cpp.printB();
    }

    MPI_Finalize();
}