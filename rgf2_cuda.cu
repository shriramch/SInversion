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

void rgf2sided_cuda(Matrix &A, Matrix &G, bool sym_mat = false,
                    bool save_off_diag = true) {
    int processRank, blockSize, matrixSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    int nblocks = matrixSize / blockSize; // assume divisible
    int nblocks_2 = nblocks / 2;          // assume divisible

    if (processRank == 0) {
        rgf2sided_upperprocess(A, G, nblocks_2, sym_mat, save_off_diag);

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
        rgf2sided_lowerprocess(A, G_dup, nblocks - nblocks_2, sym_mat,
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

void rgf2sided_upperprocess_cuda(Matrix &A, Matrix &G, int nblocks_2,
                                 bool sym_mat = false,
                                 bool save_off_diag = true) {}

void rgf2sided_lowerprocess_cuda(Matrix &A, Matrix &G, int nblocks_2,
                                 bool sym_mat = false,
                                 bool save_off_diag = true) {}

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