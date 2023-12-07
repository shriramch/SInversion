#include "rgf2_cuda.hpp"
// #include <cublas_v2.h>
#include <cuda_runtime.h>
// #include <cusolverDn.h>
#include <cuda.h>

int kernels_num_blocks, kernels_num_threads;

void kernel_init(int n) {
  kernels_num_blocks = kernels_num_threads = n;
}

__global__ void matrixSubtractKernel(float *A, float *B, float *result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        result[index] = A[index] - B[index];
    }
}

void matrixSubtracter(float *A, float *B, float *result, int n) {
  matrixSubtractKernel<<<kernels_num_blocks, kernels_num_threads>>>(A, B, result, n);
}

__global__ void matrixAddKernel(float *A, float *B, float *result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        result[index] = A[index] + B[index];
    }
}

void matrixAdder(float *A, float *B, float *result, int n) {
  matrixAddKernel<<<kernels_num_blocks, kernels_num_threads>>>(A, B, result, n);
}

__global__ void matrixScaleKernel(float *A, float k, float *result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        result[index] = A[index] * k;
    }
}

void matrixScaler(float *A, float k, float *result, int n) {
  matrixScaleKernel<<<kernels_num_blocks, kernels_num_threads>>>(A, k, result, n);
}

// void matrixMultiplyKernel(float *A, float *B, float *result, int n,
//                           cublasHandle_t cublasHandle) {
//     const float alpha = 1.0f;
//     const float beta = 0.0f;
//     cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n,
//                 B, n, &beta, result, n);
// }

// void matrixInversionKernel(float *A, float *result, int n,
//                            cusolverDnHandle_t cusolverHandle) {
//     float *identity_matrix = createIdentityMatrix(n);
//     int *d_info = nullptr; /* error info */
//     cudaMalloc(&d_info, sizeof(int));

//     // Create a temporary matrix on the device
//     float *d_A, *d_identity, *d_work;
//     cudaMalloc(&d_A, n * n * sizeof(float));
//     cudaMalloc(&d_identity, n * n * sizeof(float));
//     cudaMalloc(&d_work, n * n * sizeof(float));
//     int *ipiv;
//     cudaMalloc(&ipiv, n * sizeof(int));

//     // Copy the input matrix A to the device
//     cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_identity, identity_matrix, n * n * sizeof(float),
//                cudaMemcpyHostToDevice);

//     // Perform LU decomposition on the device
//     cusolverDnSgetrf(cusolverHandle, n, n, d_A, n, d_work, NULL,
//                      d_info); // Not using PIVOT for now

//     // Solving AX = I  , where X is the result_matrix, and I is the
//     // identity_matrix. Since AA^(-1) = I It saves on the result_matrix
//     // (identity) the answer
//     cusolverDnSgetrs(cusolverHandle, CUBLAS_OP_N, n, n, d_A, n, NULL,
//                      d_identity, n, d_info); // Not using PIVOT for now

//     // std::cout << "printing d_identity from CUDA after cusolverDnSgetrs: \n";
//     // printFloatArrayFromCuda(d_identity, n * n);
//     cudaMemcpy(result, d_identity, n * n * sizeof(float),
//                cudaMemcpyDeviceToHost);

//     // Clean up
//     free(identity_matrix);
//     cudaFree(d_A);
//     cudaFree(d_work);
//     cudaFree(ipiv);
//     cudaFree(d_identity);
//     cudaFree(d_info);
// }

// void matrixTransposeKernel(const float *A, float *result, int n,
//                            cublasHandle_t cublasHandle) {
//     const float alpha = 1.0f;
//     const float beta = 0.0f;

//     // Allocate device memory for the input and output matrices
//     float *d_A, *d_result;
//     cudaMalloc((void **)&d_A, n * n * sizeof(float));
//     cudaMalloc((void **)&d_result, n * n * sizeof(float));

//     // Copy the input matrix A to device memory
//     cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);

//     // Perform the transposition
//     cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, d_A, n,
//                 &beta, NULL, n, d_result, n);

//     // Copy the transposed matrix back to the host memory
//     cudaMemcpy(result, d_result, n * n * sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_A);
//     cudaFree(d_result);
// }

// void rgf2sided_upperprocess_cuda_prempi(Matrix &input_A, Matrix &input_G, int nblocks_2,
//                                  bool sym_mat,
//                                  bool save_off_diag, float *send);

// void rgf2sided_upperprocess_cuda_postmpi(Matrix &input_A, Matrix &input_G, int nblocks_2,
//                                  bool sym_mat,
//                                  bool save_off_diag, float *recv);

// void rgf2sided_lowerprocess_cuda_prempi(Matrix &input_A, Matrix &input_G, int nblocks_2,
//                                  bool sym_mat,
//                                  bool save_off_diag);

// void rgf2sided_lowerprocess_cuda_midmpi(Matrix &input_A, Matrix &input_G, int nblocks_2,
//                                  bool sym_mat,
//                                  bool save_off_diag);

// void rgf2sided_lowerprocess_cuda_postmpi(Matrix &input_A, Matrix &input_G, int nblocks_2,
//                                  bool sym_mat,
//                                  bool save_off_diag);