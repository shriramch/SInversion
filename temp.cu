#include <cusolverDn.h>
#include <iostream>

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

class Matrix {
public:
    static void invCUDA(int n, const float *A, float *result);
    static void matrixTransposeKernel(const float* A, float* result, int n);
};

int main() {
    // Test Inverse
    const int n = 4;
    const float A[] = {1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4};
    float result[n * n];

    Matrix::invCUDA(n, A, result);

    // Print the result
    std::cout << "Inverse Matrix:" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << result[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    // Test Transpose
    const float A_transpose[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float result_transpose[n * n];

    Matrix::matrixTransposeKernel(A_transpose, result_transpose, n);

    // Print the result
    std::cout << "Transpose Matrix:" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << result_transpose[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

// NOT WORKING .-.
void Matrix::matrixTransposeKernel(const float* A, float* result, int n) {
    
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Transpose A and store the result in 'result'
    cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, A, n, &beta, A, n, result, n);

    cublasDestroy(cublasHandle);
}

// Helper github with example (there are also streams): https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/getrf/cusolver_getrf_example.cu
// Note:
// LDA (Leading Dimension of A): In a matrix multiplication involving matrix A, the leading dimension (LDA) is the number of elements in the first dimension of A (usually the number of rows). It is the stride used to access the next column of A.
// LDB (Leading Dimension of B): In a matrix multiplication involving matrix B, the leading dimension (LDB) is the number of elements in the first dimension of B (usually the number of rows). It is the stride used to access the next column of B.
// info (or devInfo): After the function executes, the value of 'info' can be checked to determine the success or failure of the operation. A value of 0 usually indicates successful execution, while non-zero values or specific values may indicate different types of errors or warnings.
void Matrix::invCUDA(int n, const float *A, float *result) {
    // Create the Identity matrix, TODO, just make it directly on GPU if possible
    float *identity_matrix = createIdentityMatrix(n);
    int *d_info = nullptr; /* error info */

    // Handle
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    // Create a temporary matrix on the device
    float *d_A, *d_identity, *d_work;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_identity, n * n * sizeof(float));
    cudaMalloc(&d_work, n * n * sizeof(float));

    // Copy the input matrix A to the device
    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_identity, identity_matrix, n * n * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "printing d_A from CUDA: \n"; 
    printFloatArrayFromCuda(d_A, n * n);
    std::cout << "printing d_identity from CUDA: \n"; 
    printFloatArrayFromCuda(d_identity, n * n);

    // Perform LU decomposition on the device
    int *ipiv;
    cudaMalloc(&ipiv, n * sizeof(int));
    cusolverDnSgetrf(handle, n, n, d_A, n, d_work, NULL, d_info); // Not using PIVOT for now
    
    std::cout << "printing d_A from CUDA after cusolverDnSgetrf: \n"; 
    printFloatArrayFromCuda(d_A, n * n);

    // Solve for each column of the identity matrix to obtain the inverse
    // It saves on the result_matrix (identity) the answer
    cusolverDnSgetrs(handle, CUBLAS_OP_N, n, n, d_A, n, NULL, d_identity, n, d_info); // Not using PIVOT for now
    
    std::cout << "printing d_identity from CUDA after cusolverDnSgetrs: \n"; 
    printFloatArrayFromCuda(d_identity, n * n);

    cudaMemcpy(result, d_identity, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_work);
    cudaFree(ipiv);
    cudaFree(d_identity);

    cusolverDnDestroy(handle);
}
