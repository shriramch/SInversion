#include <cuda_runtime.h>
#include "rgf1_cuda.hpp" // TODO, uncomment once figured DaVinci sudo problem
#include <cublas_v2.h>
// #include <cublas.h>
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
void matrixInversionKernel(float *A, float *result, int n, cublasHandle_t cublasHandle) {
    // Code inspired by this stackoverflow https://stackoverflow.com/questions/37731103/cublas-matrix-inverse-much-slower-than-matlab
    // NOTE as mentioned the function cublasSgetriBatched is for SMALL matrixes:
    //      "This function is intended to be used for matrices of small sizes where the launch overhead is a significant factor."
    // cit. https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-getribatched

    // TODO, for now comment it out, NOTE that the inversion function is wrong as it expect a float**  (=matrix and NOT an array, which is what we are passing)
    // int* dLUPivots; // Pivoting array
    // int* dLUInfo; // Device array to store inversion status
    // int batchSize = 1; // Assuming a single matrix inversion

    // cudaMalloc(&dLUPivots, n * sizeof(int));
    // cudaMalloc(&dLUInfo, sizeof(int));

    // cublasSgetrfBatched(cublasHandle, n, &A, n, dLUPivots, dLUInfo, batchSize); // RIP does not work, NOTE it expect a float** NOT a float*
    // cudaDeviceSynchronize(); // TODO, not sure i need this sync the previous kernel

    // cublasSgetriBatched(cublasHandle, n, &A, n, dLUPivots, &result, n, dLUInfo, batchSize);
    // cudaDeviceSynchronize(); // TODO, not sure i need this sync the previous kernel

    // cudaFree(dLUPivots);
//     // cudaFree(dLUInfo);

/// STUFF FROM CHATGPT that is NOT working lol
// V1
// // Create a temporary matrix on the device
//     // float *d_A, *d_result;
//     // cudaMalloc(&d_A, n * n * sizeof(float));
//     // cudaMalloc(&d_result, n * n * sizeof(float));
//     // cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);

//     // // Allocate memory for the pivot array on the device
//     // int *d_ipiv;
//     // cudaMalloc(&d_ipiv, n * sizeof(int));

//     // // Perform LU decomposition
//     // cublasSgetrf(cublasHandle, n, n, d_A, n, d_ipiv);

//     // // Perform matrix inversion using the LU decomposition
//     // cublasSgetri(cublasHandle, n, d_A, n, d_ipiv, d_result, n);

//     // // Copy the inverted matrix back to the host memory
//     // cudaMemcpy(result, d_result, n * n * sizeof(float), cudaMemcpyDeviceToHost);

//     // // Free the allocated memory on the device
//     // cudaFree(d_A);
//     // cudaFree(d_result);
//     // cudaFree(d_ipiv);


// V2
// // Create a temporary matrix on the device
//     cusolverDnHandle_t handle;
//     cusolverDnCreate(&handle);

//     float *d_A, *d_result;
//     cudaMalloc(&d_A, n * n * sizeof(float));
//     cudaMalloc(&d_result, n * n * sizeof(float));

//     // Copy the input matrix A to the device
//     cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);

//     // Perform LU decomposition on the device
//     int *ipiv;
//     cudaMalloc(&ipiv, n * sizeof(int));
//     cusolverDnSgetrf(handle, n, n, d_A, n, d_result, ipiv, nullptr);

//     // Perform matrix inversion on the device
//     cusolverDnSgetri(handle, n, d_result, n, ipiv, nullptr, 0, nullptr);

//     // Copy the result back to the host
//     cudaMemcpy(result, d_result, n * n * sizeof(float), cudaMemcpyDeviceToHost);

//     // Clean up
//     cudaFree(d_A);
//     cudaFree(d_result);
//     cudaFree(ipiv);
//     cusolverDnDestroy(handle);

// V3
    // int lda = n;
    // int *devInfo = nullptr;
    // float *d_A = nullptr;
    // int lwork = 0;
    // float one = 1.0f;

    // cudaMalloc((void**)&devInfo, sizeof(int));
    // cudaMalloc((void**)&d_A, sizeof(float) * n * n);

    // // Copy input matrix to device
    // cudaMemcpy(d_A, A, sizeof(float) * n * n, cudaMemcpyHostToDevice);

    // // Create a handle for cuSolver
    // cusolverDnHandle_t cusolverHandle;
    // cusolverDnCreate(&cusolverHandle);

    // // LU factorization workspace size query
    // cusolverDnSgetrf_bufferSize(cusolverHandle, n, n, d_A, lda, &lwork);

    // // Perform LU factorization
    // cudaMalloc((void**)&d_A, sizeof(float) * n * n);
    // int *d_ipiv = nullptr;
    // cudaMalloc((void**)&d_ipiv, sizeof(int) * n);
    // cusolverDnSgetrf(cusolverHandle, n, n, d_A, lda, nullptr, d_ipiv, devInfo);

    // // Check for factorization success
    // int devInfo_h = 0;
    // cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

    // if (devInfo_h != 0) {
    //     std::cerr << "Factorization failed: Matrix is singular." << std::endl;
    //     return;
    // }

    // // Solve the system of linear equations for each column of the identity matrix
    // int *d_pivot = nullptr;
    // cudaMalloc((void**)&d_pivot, sizeof(int) * n);
    // float *d_identity = nullptr;
    // cudaMalloc((void**)&d_identity, sizeof(float) * n * n);
    // cudaMemcpy(d_identity, result, sizeof(float) * n * n, cudaMemcpyHostToDevice);

    // cublasStrsm(cublasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
    //         CUBLAS_DIAG_UNIT, n, n, reinterpret_cast<const float*>(&one), d_A, lda, d_identity, n);


    // cublasStrsm(cublasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
    //         CUBLAS_DIAG_NON_UNIT, n, n, reinterpret_cast<const float*>(&one), d_A, lda, d_identity, n);


    // // Copy the result back to the host
    // cudaMemcpy(result, d_identity, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

    // // Clean up
    // cudaFree(devInfo);
    // cudaFree(d_A);
    // cudaFree(d_pivot);
    // cudaFree(d_identity);
    // cusolverDnDestroy(cusolverHandle);
}

// TODO, test this function
void matrixTransposeKernel(const float* A, float* result, int n, cublasHandle_t cublasHandle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Transpose A and store the result in 'result'
    cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, A, n, &beta, A, n, result, n);
}

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
// CUDA-accelerated rgf1sided function

void rgf1sided_cuda(Matrix &input_A, Matrix &input_G, bool sym_mat, bool save_off_diag) {
    printf("Starting rgf1sided_cuda!\n");
    int blockSize, matrixSize;
    input_A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    printf("input_A.getBlockSizeAndMatrixSize(blockSize, matrixSize);!\n");
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
    matrixInversionKernel(A_mdiag, G_mdiag, blockSize, cublasHandle); // TODO, Does not do anything
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
            (AGi, &(G_mdiag[i * blockSize * blockSize]), blockSize, cublasHandle);

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


    inputMatrix.printM();
    tempResult.printM();

}