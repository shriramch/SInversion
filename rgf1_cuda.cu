#include "argparse.h"
#include "rgf1.hpp"
#include "rgf1_cuda.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

float *identity_matrix;
int *d_info = nullptr;
float *d_A, *d_identity, *d_work;
int *ipiv;
float *d_result;

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

// void createIdentityMatrix(float *A, int n) {
//     // float *identityMatrix = (float *)malloc(n * n * sizeof(float));
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             int index = i * n + j;
//             A[index] = (i == j) ? 1 : 0;
//         }
//     }
// }

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

// void matrixTransposeKernel(const float *A, float *result, int n,
//                            cublasHandle_t cublasHandle) {
//     const float alpha = 1.0f;
//     const float beta = 0.0f;

//     // Copy the input matrix A to device memory
//     cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);

//     // Perform the transposition
//     cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, d_A, n,
//                 &beta, NULL, n, d_result, n);

//     // Copy the transposed matrix back to the host memory
//     cudaMemcpy(result, d_result, n * n * sizeof(float),
//     cudaMemcpyDeviceToHost);

//     // Free device memory
// }

void matrixTransposeKernel(const float *A, float *result, int n,
                           cublasHandle_t cublasHandle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Copy the input matrix A to device memory
    // cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Perform the transposition
    cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, A, n,
                &beta, NULL, n, result, n);

    // Copy the transposed matrix back to the host memory
    // cudaMemcpy(result, d_result, n * n * sizeof(float),
    // cudaMemcpyDeviceToHost);

    // Free device memory
}

// CUDA-accelerated rgf1sided function
void rgf1sided_cuda(Matrix &input_A, Matrix &input_G, bool sym_mat,
                    bool save_off_diag) {
    int blockSize, matrixSize;
    input_A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    int nblocks = matrixSize / blockSize;

    int kernels_num_blocks = nblocks;
    int kernels_num_threads = nblocks;
    size_t blockSizeBytes = blockSize * blockSize * sizeof(float);

    // Initialize the handle used for cuBLAS
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;

    cublasCreate(&cublasHandle);
    cusolverDnCreate(&cusolverHandle);

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
    cudaMemcpy(A_updiag, input_A.updiag, size_updiag, cudaMemcpyHostToDevice);
    cudaMemcpy(G_updiag, input_G.updiag, size_updiag, cudaMemcpyHostToDevice);

    float *A_lodiag, *G_lodiag;
    cudaMalloc(&A_lodiag, size_updiag);
    cudaMalloc(&G_lodiag, size_updiag);

    // Copy matrices from host to device
    cudaMemcpy(A_lodiag, input_A.lodiag, size_updiag, cudaMemcpyHostToDevice);
    cudaMemcpy(G_lodiag, input_G.lodiag, size_updiag, cudaMemcpyHostToDevice);

    float *AAi, *AGi;
    cudaMalloc(&AAi, blockSizeBytes);
    cudaMalloc(&AGi, blockSizeBytes);

    // Inverse and transpose kernel variables
    cudaMalloc(&d_info, sizeof(int));
    cudaMalloc(&d_A, blockSize * blockSize * sizeof(float));
    cudaMalloc(&d_identity, blockSize * blockSize * sizeof(float));
    cudaMalloc(&d_work, blockSize * blockSize * sizeof(float));
    cudaMalloc(&ipiv, blockSize * sizeof(int));
    cudaMalloc((void **)&d_result, blockSize * blockSize * sizeof(float));

    identity_matrix = createIdentityMatrix(blockSize);

    // cudaMemcpy(d_identity, identity_matrix, blockSize * blockSize *
    // sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernels for matrix operations

    // 0. Inverse of the first block
    matrixInversionKernel(A_mdiag, G_mdiag, blockSize, cusolverHandle);
    // 1. Forward substitution (performed left to right)

    for (int i = 1; i < nblocks; ++i) {

        // cudaMemset(AAi, 0, blockSizeBytes);
        // cudaMemset(AGi, 0, blockSizeBytes);

        matrixMultiplyKernel(&(A_lodiag[(i - 1) * blockSize * blockSize]),
                             &(G_mdiag[(i - 1) * blockSize * blockSize]), AGi,
                             blockSize, cublasHandle);
        matrixMultiplyKernel(AGi, &(A_updiag[(i - 1) * blockSize * blockSize]),
                             AAi, blockSize, cublasHandle);
        matrixSubtractKernel<<<kernels_num_blocks, kernels_num_threads>>>(
            &(A_mdiag[i * blockSize * blockSize]), AAi, AGi, blockSize);
        matrixInversionKernel(AGi, &(G_mdiag[i * blockSize * blockSize]),
                              blockSize, cusolverHandle);

        // Free temporary GPU memory
    }

    // 2. Backward substitution
    float *Glf, *Glf1;
    cudaMalloc(&Glf, blockSizeBytes);
    cudaMalloc(&Glf1, blockSizeBytes);
    float *Guf, *Guf1;
    cudaMalloc(&Guf, blockSizeBytes);
    cudaMalloc(&Guf1, blockSizeBytes);

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

    // printFloatArrayFromCuda(G, matrix_array_size);

    // Copy results back to host
    // cudaMemcpy(input_A.getMat(), A, size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(input_G.getMat(), G, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_A.mdiag, A_mdiag, size_mdiag, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.mdiag, G_mdiag, size_mdiag, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_A.updiag, A_updiag, size_updiag, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.updiag, G_updiag, size_updiag, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_A.lodiag, A_lodiag, size_updiag, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_G.lodiag, G_lodiag, size_updiag, cudaMemcpyDeviceToHost);

    // Free GPU memory
    // cudaFree(A);
    // cudaFree(G);
    cudaFree(A_mdiag);
    cudaFree(G_mdiag);
    cudaFree(A_updiag);
    cudaFree(G_updiag);
    cudaFree(A_lodiag);
    cudaFree(G_lodiag);
    cudaFree(AAi);
    cudaFree(AGi);
    // Free temporary GPU memory
    cudaFree(Guf);
    cudaFree(Guf1);
    // Free temporary GPU memory
    cudaFree(Glf);
    cudaFree(Glf1);

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

// typedef struct {
//     int matrixSize;
//     int blockSize;
//     int numRuns;
//     bool isSymmetric;
//     bool saveOffDiag;
//     char *inputPath;
// } Config;

// void InitOptions(Config *config) {
//     config->blockSize = 2;
//     config->matrixSize = 0;
//     config->numRuns = 10;
//     config->isSymmetric = false;
//     config->saveOffDiag = true;
//     config->inputPath = NULL;
// }

// int parse(Config *config, int argc, const char **argv) {
//     static const char *const usages[] = {
//         NULL,
//     };
//     struct argparse_option options[] = {
//         OPT_HELP(),
//         OPT_INTEGER('m', "matrixSize", &config->matrixSize, "matrix size",
//         NULL,
//                     0, 0),
//         OPT_INTEGER('b', "blockSize", &config->blockSize, "block size", NULL,
//         0,
//                     0),
//         OPT_INTEGER('n', "numRuns", &config->numRuns, "number of runs", NULL,
//         0,
//                     0),
//         OPT_INTEGER('s', "isSymmetric", &config->isSymmetric, "is symmetric",
//                     NULL, 0, 0),
//         OPT_INTEGER('o', "saveOffDiag", &config->saveOffDiag, "save off
//         diag", NULL, 0, 0), OPT_STRING('f', "inputPath", &config->inputPath,
//         "input path", NULL,0,0), OPT_END(),
//     };

//     struct argparse argparse;
//     argparse_init(&argparse, options, usages, 0);
//     argparse_describe(&argparse, "DPHPC TEAM", NULL);
//     argc = argparse_parse(&argparse, argc, argv);

//     return 0;
// }

// int main(int argc, const char *argv[]) {
//     const char *bin_name = argv[0];
//     Config config;
//     InitOptions(&config);
//     parse(&config, argc, argv);
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

//         // inputMatrix.printB();
//         Matrix tempResult(
//             MATRIX_SIZE); // zero initialization, same shape as inputMatrix
//         tempResult.convertDenseToBlkTridiag(
//             BLOCK_SIZE); // G has same blockSize as inputMatrix
//         rgf1sided_cuda(inputMatrix, tempResult, IS_SYMMETRIC, SAVE_OFF_DIAG);

//         tempResult.printB();
//         std::cout << "\n########################################## \n";

//         // inputMatrix.printB();
//         // Check against the already implemented RGF1 on C++
//         Matrix tempResult_cpp(
//             MATRIX_SIZE); // zero initialization, same shape as inputMatrix
//         tempResult_cpp.convertDenseToBlkTridiag(
//             BLOCK_SIZE); // G has same blockSize as inputMatrix

//         rgf1sided(inputMatrix, tempResult_cpp, IS_SYMMETRIC, SAVE_OFF_DIAG);

//         tempResult_cpp.printB();

//     }
// }