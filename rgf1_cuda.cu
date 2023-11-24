#include "rgf1.hpp"

#include <cuda_runtime.h>
#include "rgf1_cuda.hpp" // TODO, uncomment once figured DaVinci sudo problem
#include <cublas_v2.h>
// #include <cublas.h>
#include <cusolverDn.h>

// TODO, not that the cuSolver we are using now is the Dense, check whether we have to use the sparse one

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

// Function to create an identity matrix of size n x n
float* createIdentityMatrix(int n) {
    float* identityMatrix = (float*) malloc(n * n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int index = i * n + j;
            identityMatrix[index] = (i == j) ? 1 : 0;
        }
    }
    return identityMatrix;
}

// TODO, test this function
void matrixInversionKernel(float *A, float *result, int n, cusolverDnHandle_t cusolverHandle) {
    // Create the Identity matrix, TODO, just make it directly on GPU if possible
    float *identity_matrix = createIdentityMatrix(n);
    int *d_info = nullptr; /* error info */

    // Create a temporary matrix on the device
    float *d_A, *d_identity, *d_work;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_identity, n * n * sizeof(float));
    cudaMalloc(&d_work, n * n * sizeof(float));
    int *ipiv;
    cudaMalloc(&ipiv, n * sizeof(int));

    // Copy the input matrix A to the device
    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_identity, identity_matrix, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Perform LU decomposition on the device
    cusolverDnSgetrf(cusolverHandle, n, n, d_A, n, d_work, NULL, d_info); // Not using PIVOT for now

    // Solving AX = I  , where X is the result_matrix, and I is the identity_matrix. Since AA^(-1) = I
    // It saves on the result_matrix (identity) the answer
    cusolverDnSgetrs(cusolverHandle, CUBLAS_OP_N, n, n, d_A, n, NULL, d_identity, n, d_info); // Not using PIVOT for now
    
    std::cout << "printing d_identity from CUDA after cusolverDnSgetrs: \n"; 
    printFloatArrayFromCuda(d_identity, n * n);
    cudaMemcpy(result, d_identity, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_identity);
    cudaFree(d_work);
    cudaFree(ipiv);
}


// TODO, test this function
void matrixTransposeKernel(const float* A, float* result, int n, cublasHandle_t cublasHandle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Transpose A and store the result in 'result'
    cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, A, n, &beta, A, n, result, n);
}


// CUDA-accelerated rgf1sided function

void rgf1sided_cuda(Matrix &input_A, Matrix &input_G, bool sym_mat, bool save_off_diag) {
    printf("Starting rgf1sided_cuda!\n");
    int blockSize, matrixSize;
    input_A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    printf("input_A.getBlockSizeAndMatrixSize(blockSize, matrixSize);!\n");
    int nblocks = matrixSize / blockSize;
    

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
    cudaMemcpy(G, input_G.getMat(), size, cudaMemcpyHostToDevice); // Actually not really need to copy now tho
    std::cout << "printing A: \n"; 
    printFloatArray(input_A.getMat(), matrix_array_size);
    std::cout << "printing G: \n"; 
    printFloatArray(input_G.getMat(), matrix_array_size);
    std::cout << "printing A from CUDA: \n"; 
    printFloatArrayFromCuda(A, matrix_array_size);
    std::cout << "printing G from CUDA: \n"; 
    printFloatArrayFromCuda(G, matrix_array_size);

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
    matrixInversionKernel(A_mdiag, G_mdiag, blockSize, cusolverHandle); // TODO, Does not do anything
    std::cout << "After 0. Inverse of the first block\n";
    std::cout << "printing A_mdiag from CUDA: \n"; 
    printFloatArrayFromCuda(A_mdiag, size_mdiag / sizeof(float));
    std::cout << "printing G_mdiag from CUDA: \n"; 
    printFloatArrayFromCuda(G_mdiag, size_mdiag / sizeof(float));

    int kernels_num_blocks = nblocks; // TODO, find the optimal combination
    int kernels_num_threads = 1; // TODO, find the optimal combination
    
    // 1. Forward substitution (performed left to right)
    std::cout << "Before 1. Forward substitution (performed left to right)\n";
    std::cout << "printing A_updiag from CUDA: \n"; 
    printFloatArrayFromCuda(A_updiag, size_updiag / sizeof(float));
    std::cout << "printing A_mdiag from CUDA: \n"; 
    printFloatArrayFromCuda(A_mdiag, size_mdiag / sizeof(float));
    std::cout << "printing A_lodiag from CUDA: \n"; 
    printFloatArrayFromCuda(A_lodiag, size_updiag / sizeof(float));
    std::cout << "printing G_mdiag from CUDA: \n"; 
    printFloatArrayFromCuda(G_mdiag, size_mdiag / sizeof(float));
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
        matrixInversionKernel
            (AGi, &(G_mdiag[i * blockSize * blockSize]), blockSize, cusolverHandle);

        // Free temporary GPU memory
        cudaFree(AAi);
        cudaFree(AGi);
    }
    std::cout << "After 1. Forward substitution (performed left to right)\n";
    std::cout << "printing A_updiag from CUDA: \n"; 
    printFloatArrayFromCuda(A_updiag, size_updiag / sizeof(float));
    std::cout << "printing A_mdiag from CUDA: \n"; 
    printFloatArrayFromCuda(A_mdiag, size_mdiag / sizeof(float));
    std::cout << "printing A_lodiag from CUDA: \n"; 
    printFloatArrayFromCuda(A_lodiag, size_updiag / sizeof(float));
    std::cout << "printing G_mdiag from CUDA: \n"; 
    printFloatArrayFromCuda(G_mdiag, size_mdiag / sizeof(float));
    std::cout << "----------------------------------------------------------------------- \n"; 


    // 2. Backward substitution
    std::cout << "Before 2. Backward substitution\n";
    std::cout << "printing A_updiag from CUDA: \n"; 
    printFloatArrayFromCuda(A_updiag, size_updiag / sizeof(float));
    std::cout << "printing A_mdiag from CUDA: \n"; 
    printFloatArrayFromCuda(A_mdiag, size_mdiag / sizeof(float));
    std::cout << "printing A_lodiag from CUDA: \n"; 
    printFloatArrayFromCuda(A_lodiag, size_updiag / sizeof(float));
    std::cout << "printing G_updiag from CUDA: \n"; 
    printFloatArrayFromCuda(G_updiag, size_updiag / sizeof(float));
    std::cout << "printing G_mdiag from CUDA: \n"; 
    printFloatArrayFromCuda(G_mdiag, size_mdiag / sizeof(float));
    std::cout << "printing G_lodiag from CUDA: \n"; 
    printFloatArrayFromCuda(G_lodiag, size_updiag / sizeof(float));
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
                matrixTransposeKernel
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
    std::cout << "After 2. Backward substitution\n";
    std::cout << "printing A_updiag from CUDA: \n"; 
    printFloatArrayFromCuda(A_updiag, size_updiag / sizeof(float));
    std::cout << "printing A_mdiag from CUDA: \n"; 
    printFloatArrayFromCuda(A_mdiag, size_mdiag / sizeof(float));
    std::cout << "printing A_lodiag from CUDA: \n"; 
    printFloatArrayFromCuda(A_lodiag, size_updiag / sizeof(float));
    std::cout << "printing G_updiag from CUDA: \n"; 
    printFloatArrayFromCuda(G_updiag, size_updiag / sizeof(float));
    std::cout << "printing G_mdiag from CUDA: \n"; 
    printFloatArrayFromCuda(G_mdiag, size_mdiag / sizeof(float));
    std::cout << "printing G_lodiag from CUDA: \n"; 
    printFloatArrayFromCuda(G_lodiag, size_updiag / sizeof(float));
    std::cout << "----------------------------------------------------------------------- \n";

    std::cout << "printing A from CUDA: \n"; 
    printFloatArrayFromCuda(A, matrix_array_size);
    std::cout << "printing G from CUDA: \n"; 
    printFloatArrayFromCuda(G, matrix_array_size);

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
    cusolverDnDestroy(cusolverHandle);
}

// TEMP main to test stuff out
int main(int argc, const char *argv[]) {
    
    int MATRIX_SIZE = 4;
    int BLOCK_SIZE = 2;
    bool IS_SYMMETRIC = false;
    bool SAVE_OFF_DIAG = true;

    // Matrix inputMatrix = generateBandedDiagonalMatrix(MATRIX_SIZE, 2, true, 0);
    Matrix inputMatrix = generateFixedMatrixOfSize4();
    inputMatrix.convertDenseToBlkTridiag(BLOCK_SIZE);

    Matrix tempResult(MATRIX_SIZE); // zero initialization, same shape as inputMatrix
    tempResult.convertDenseToBlkTridiag( BLOCK_SIZE); // G has same blockSize as inputMatrix
    rgf1sided_cuda(inputMatrix, tempResult, IS_SYMMETRIC, SAVE_OFF_DIAG);

    inputMatrix.printB();
    tempResult.printB();
    std::cout << "##########################################/n"; 

    // Check against the already implemented RGF1 on C++
    Matrix tempResult_cpp(MATRIX_SIZE); // zero initialization, same shape as inputMatrix
    tempResult_cpp.convertDenseToBlkTridiag( BLOCK_SIZE); // G has same blockSize as inputMatrix
    
    rgf1sided(inputMatrix, tempResult_cpp, false, true);

    inputMatrix.printB();
    tempResult_cpp.printB();


}