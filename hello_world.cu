#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// Error checking for CUDA
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
template <typename T>
void check_cuda(T err, const char *const func, const char *const file,
                int const line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// Error checking for cuBLAS
#define checkCublasErrors(val) check_cublas((val), #val, __FILE__, __LINE__)
template <typename T>
void check_cublas(T err, const char *const func, const char *const file,
                  int const line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error at: " << file << ":" << line << std::endl;
        std::cerr << "Error code: " << err << " " << func << std::endl;
        exit(1);
    }
}

// Print the matrix
void printMatrix(const float *matrix, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

// Main function
int main() {
    const int N = 3; // Matrix size

    cublasHandle_t handle;
    checkCublasErrors(cublasCreate(&handle));

    // Allocate and initialize host matrices
    float A[N * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float B[N * N] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    float C[N * N] = {0};

    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void **)&d_A, N * N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_B, N * N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_C, N * N * sizeof(float)));

    // Copy matrices from the host to  device
    checkCudaErrors(
        cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice));

    // Perform matrix multiplication: C = A * B
    const float alpha = 1.0f;
    const float beta = 0.0f;
    checkCublasErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                  &alpha, d_A, N, d_B, N, &beta, d_C, N));

    // Copy  result back to  host
    checkCudaErrors(
        cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print  result
    std::cout << "Result Matrix C:" << std::endl;
    printMatrix(C, N, N);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}