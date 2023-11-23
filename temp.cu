#include <cusolverDn.h>
#include <iostream>

class Matrix {
public:
    static void invCUDA(int n, const float *A, float *result);
};

int main() {
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

    return 0;
}

void Matrix::invCUDA(int n, const float *A, float *result) {
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    // Create a temporary matrix on the device
    float *d_A, *d_result;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_result, n * n * sizeof(float));

    // Copy the input matrix A to the device
    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Perform LU decomposition on the device
    int *ipiv;
    cudaMalloc(&ipiv, n * sizeof(int));
    cusolverDnSgetrf(handle, n, n, d_A, n, d_result, ipiv, nullptr);

    // Solve for each column of the identity matrix to obtain the inverse
    float *d_identity;
    cudaMalloc(&d_identity, n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        cudaMemcpy(d_identity, &A[i], n * sizeof(float), cudaMemcpyHostToDevice);
        cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, d_result, n, ipiv, d_identity, n, nullptr);
        cudaMemcpy(&result[i * n], d_identity, n * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_result);
    cudaFree(ipiv);
    cudaFree(d_identity);

    cusolverDnDestroy(handle);
}
