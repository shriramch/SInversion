#include "matrices_utils.hpp"
// memory safety -- take care

Matrix::~Matrix() {
    delete[] mat;
    delete[] mdiag;
    delete[] updiag;
    delete[] lodiag;
}

void Matrix::copyMatrixData(const Matrix &other) {
    // deep copy
    matrixSize = other.matrixSize;
    blockSize = other.blockSize;
    int nblocks = matrixSize / blockSize;
    mat = new float[matrixSize * matrixSize];
    mdiag = new float[nblocks * blockSize * blockSize];
    updiag = new float[(nblocks - 1) * blockSize * blockSize];
    lodiag = new float[(nblocks - 1) * blockSize * blockSize];
    memcpy(mat, other.mat, matrixSize * matrixSize * sizeof(float));
    memcpy(mdiag, other.mdiag, nblocks * blockSize * blockSize * sizeof(float));
    memcpy(updiag, other.updiag,
           (nblocks - 1) * blockSize * blockSize * sizeof(float));
    memcpy(lodiag, other.lodiag,
           (nblocks - 1) * blockSize * blockSize * sizeof(float));
}

// Copy constructor
Matrix::Matrix(const Matrix &other) { copyMatrixData(other); }
// Copy assignment operator - allow different size
Matrix &Matrix::operator=(const Matrix &other) {
    if (this != &other) {
        delete[] mat;
        delete[] mdiag;
        delete[] updiag;
        delete[] lodiag;
        copyMatrixData(other);
    }
    return *this;
}

/* Zero-initialize matrix  */
Matrix::Matrix(int N) {
    matrixSize = N;
    // mat = new float[N * N]();
    blockSize = 1;
    mdiag = NULL;
    updiag = NULL;
    lodiag = NULL;
}

/* initlize matrix with size N and values newMat */
Matrix::Matrix(int N, float *newMat) {
    matrixSize = N;
    // mat = new float[N * N];
    // memcpy(mat, newMat, N * N * sizeof(float));
    blockSize = 1;
    mdiag = NULL;
    updiag = NULL;
    lodiag = NULL;
}

// convert back to show the result
void Matrix::convertBlkTridiagToDense() {
    mat = new float[matrixSize * matrixSize];
    assert(matrixSize % blockSize ==
           0); // matrixSize must be divisible by blockSize
    int nblocks = matrixSize / blockSize;
    // Assume it is called after initialization of mdiag, updiag, lodiag
    for (int b = 0; b < nblocks; ++b) {
        for (int i = 0; i < blockSize; ++i) {
            for (int j = 0; j < blockSize; ++j) {
                mat[(b * blockSize + i) * matrixSize + (b * blockSize + j)] =
                    mdiag[b * blockSize * blockSize + i * blockSize + j];
            }
        }
    }

    for (int b = 0; b < nblocks - 1; ++b) {
        for (int i = 0; i < blockSize; ++i) {
            for (int j = 0; j < blockSize; ++j) {
                mat[(b * blockSize + i) * matrixSize +
                    ((b + 1) * blockSize + j)] =
                    updiag[b * blockSize * blockSize + i * blockSize + j];
                mat[((b + 1) * blockSize + i) * matrixSize +
                    (b * blockSize + j)] =
                    lodiag[b * blockSize * blockSize + i * blockSize + j];
            }
        }
    }
}

// void Matrix::transposeBLAS(int n, float *A, float *result) {
//     cblas_somatcopy(CblasRowMajor, CblasTrans, n, n, 1.0f, A, n, result, n);
// }

/* Generate 3 representations */
void Matrix::convertDenseToBlkTridiag(const int blockSize) {
    this->blockSize = blockSize;
    assert(matrixSize % blockSize ==
           0); // matrixSize must be divisible by blockSize
    int nblocks = matrixSize / blockSize;
    mdiag = new float[nblocks * blockSize * blockSize];
    updiag = new float[(nblocks - 1) * blockSize * blockSize];
    lodiag = new float[(nblocks - 1) * blockSize * blockSize];
    // for (int b = 0; b < nblocks; ++b) {
    //     for (int i = 0; i < blockSize; ++i) {
    //         for (int j = 0; j < blockSize; ++j) {
    //             mdiag[b * blockSize * blockSize + i * blockSize + j] =
    //                 mat[(b * blockSize + i) * matrixSize + (b * blockSize + j)];
    //         }
    //     }
    // }

    // for (int b = 0; b < nblocks - 1; ++b) {
    //     for (int i = 0; i < blockSize; ++i) {
    //         for (int j = 0; j < blockSize; ++j) {
    //             updiag[b * blockSize * blockSize + i * blockSize + j] =
    //                 mat[(b * blockSize + i) * matrixSize +
    //                     ((b + 1) * blockSize + j)];
    //             lodiag[b * blockSize * blockSize + i * blockSize + j] =
    //                 mat[((b + 1) * blockSize + i) * matrixSize +
    //                     (b * blockSize + j)];
    //         }
    //     }
    // }
}

void Matrix::printM() {
    std::cout << "Matrix: " << std::endl;
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            std::cout << mat[i * matrixSize + j] << " ";
        }
        std::cout << std::endl;
    }
}

void Matrix::printB() {
    std::cout << "Main diagonal: " << std::endl;
    for (int i = 0; i < matrixSize / blockSize; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            for (int k = 0; k < blockSize; ++k) {
                std::cout
                    << mdiag[i * blockSize * blockSize + j * blockSize + k]
                    << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "Upper diagonal: " << std::endl;
    for (int i = 0; i < matrixSize / blockSize - 1; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            for (int k = 0; k < blockSize; ++k) {
                std::cout
                    << updiag[i * blockSize * blockSize + j * blockSize + k]
                    << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "Lower diagonal: " << std::endl;
    for (int i = 0; i < matrixSize / blockSize - 1; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            for (int k = 0; k < blockSize; ++k) {
                std::cout
                    << lodiag[i * blockSize * blockSize + j * blockSize + k]
                    << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void Matrix::getBlockSizeAndMatrixSize(int &outBlockSize, int &outMatrixSize) {
    outBlockSize = blockSize;
    outMatrixSize = matrixSize;
}

float *Matrix::getMat() { return mat; }

// Note, we expect the same size for both the lists
bool Matrix::allclose(const float *a, const float *b, std::size_t size,
                      double rtol, double atol, bool isPrint) {
    for (std::size_t i = 0; i < size; ++i) {
        if (isPrint) {
            std::cout << a[i] << " vs " << b[i] << std::endl;
        }
        if (std::abs(a[i] - b[i]) > (atol + rtol * std::abs(b[i]))) {
            printf("Not equal value: a[i]: %f, b[i]: %f, absolute difference: "
                   "%f, relative difference: %f%%\n",
                   a[i], b[i], std::abs(a[i] - b[i]),
                   std::abs(a[i] - b[i]) / std::abs(b[i]) * 100);
            return false; // Elements are not almost equal
        }
    }

    return true; // All elements are almost equal
}

bool Matrix::compareDiagonals(const Matrix &other, bool isPrint = true) {
    // Compare main diagonal (mdiag)
    double rtol = 1e-3, atol = 1e-5;
    if (!allclose(mdiag, other.mdiag, matrixSize * blockSize, rtol, atol,
                  isPrint)) {
        std::cout << "Main diagonal not equal." << std::endl;
        return false;
    }

    // Compare upper diagonal (updiag)
    if (!allclose(updiag, other.updiag, (matrixSize - blockSize) * blockSize,
                  rtol, atol, isPrint)) {
        std::cout << "Upper diagonal not equal." << std::endl;
        return false;
    }

    // Compare lower diagonal (lodiag)
    if (!allclose(lodiag, other.lodiag, (matrixSize - blockSize) * blockSize,
                  rtol, atol, isPrint)) {
        std::cout << "Lower diagonal not equal." << std::endl;
        return false;
    }
    return true;
}

// void Matrix::mmmBLAS(int n, float *A, float *B, float *result) {
//     const CBLAS_ORDER order = CblasRowMajor;
//     const CBLAS_TRANSPOSE transA = CblasNoTrans;
//     const CBLAS_TRANSPOSE transB = CblasNoTrans;
//     const float alpha = 1.0;
//     const float beta = 0.0;

//     cblas_sgemm(order, transA, transB, n, n, n, alpha, A, n, B, n, beta, result,
//                 n);
// }

void Matrix::mmSub(int n, float *A, float *B, float *result) {
    for (int i = 0; i < n * n; ++i) {
        result[i] = A[i] - B[i];
    }
}

void Matrix::mmAdd(int n, float *A, float *B, float *result) {
    for (int i = 0; i < n * n; ++i) {
        result[i] = A[i] + B[i];
    }
}

// void Matrix::invBLAS(int n, const float *A, float *result) {

//     int *ipiv = (int *)malloc(n * sizeof(int));
//     memcpy(result, A, n * n * sizeof(float));

//     LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, result, n, ipiv);

//     LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, result, n, ipiv);

//     free(ipiv);
// }

void Matrix::matScale(int n, float *A, int k, float *result) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i * n + j] = A[i * n + j] * k;
        }
    }
}

/*
    Generate a dense matrix of shape: (matrice_size x matrice_size) filled
    with random numbers. The matrice should real valued.
*/
// Matrix generateRandomMat(int matrixSize, bool isSymmetric, int seed) {
//     Matrix mat(matrixSize);
//     float *matrix = mat.getMat();
//     if (seed != 0) {
//         // Seed the random number generator if a seed is provided
//         std::srand(seed);
//     } else {
//         // Seed the random number generator with the current time if no seed is
//         // provided
//         std::srand(static_cast<unsigned int>(std::time(nullptr)));
//     }

//     for (int i = 0; i < matrixSize; i++) {
//         for (int j = 0; j < matrixSize; j++) {
//             // Generate a random real-valued number between 0 and 1
//             float random_number =
//                 static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

//             // Store the number in the matrix
//             matrix[i * matrixSize + j] = random_number;

//             if (isSymmetric && i != j) {
//                 // If the matrix is symmetric, set the corresponding
//                 // off-diagonal element
//                 matrix[j * matrixSize + i] = random_number;
//             }
//         }
//     }
//     return mat;
// }

void generateRandomMat(int matrixSize, bool isSymmetric, float *matrix) {
    // Matrix mat(matrixSize);
    // float *matrix = mat.getMat();

    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            // Generate a random real-valued number between 0 and 1
            float random_number =
                static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

            // Store the number in the matrix
            matrix[i * matrixSize + j] = random_number;

            if (isSymmetric && i != j) {
                // If the matrix is symmetric, set the corresponding
                // off-diagonal element
                matrix[j * matrixSize + i] = random_number;
            }
        }
    }
    // return mat;
}

/*
    Generate a banded diagonal matrix of shape: matrice_size^2 with a
    bandwidth = matrice_bandwidth, filled with random numbers.
*/
// Matrix generateBandedDiagonalMatrix(int matrixSize, int matriceBandwidth,
//                                     bool isSymmetric, int seed) {
//     Matrix A = generateRandomMat(matrixSize, isSymmetric, seed);
//     float *matrix = A.getMat();
//     for (int i = 0; i < matrixSize; ++i) {
//         for (int j = 0; j < matrixSize; ++j) {
//             if (i - j > matriceBandwidth || j - i > matriceBandwidth) {
//                 matrix[i * matrixSize + j] = 0;
//             }
//         }
//     }
//     return A;
// }

Matrix generateBandedDiagonalMatrix(int matrixSize, int blockSize, bool isSymmetric, int seed) {
    Matrix A(matrixSize);
    A.convertDenseToBlkTridiag(blockSize);
    int numBlocks = matrixSize / blockSize;
    if (seed != 0) {
        // Seed the random number generator if a seed is provided
        std::srand(seed);
    } else {
        // Seed the random number generator with the current time if no seed is
        // provided
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
    }

    for (int i = 0; i < numBlocks; ++i) {
        generateRandomMat(blockSize, isSymmetric, A.mdiag + i * blockSize * blockSize);
    }

    for (int i = 0; i < numBlocks - 1; ++i) {
        generateRandomMat(blockSize, false, A.updiag + i * blockSize * blockSize);
        if (isSymmetric) {
            for (int k = 0; k < blockSize; ++k) {
                for (int j = 0; j < blockSize; ++j) {
                    A.lodiag[i * blockSize * blockSize + k * blockSize + j] = A.updiag[i * blockSize * blockSize + j * blockSize + k];
                }
            }
        } else {
            generateRandomMat(blockSize, false, A.lodiag + i * blockSize * blockSize);
        }
    }
    return A;
}

void makeMatrixBanded(int matrixSize, int matriceBandwidth, float *matrix) {
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            if (i - j > matriceBandwidth || j - i > matriceBandwidth) {
                matrix[i * matrixSize + j] = 0;
            }
        }
    }
}

// fixed matrix; matrixSize = 8, blocksize = 2, bandwidth = 2
// isSymmetric = true
Matrix generateFixedMatrixOfSize8() {
    float given_matrix[8][8] = {{0.11214, 0.976047, 0, 0, 0, 0, 0, 0},
                                {0.976047, 0.423681, 0.434601, 0, 0, 0, 0, 0},
                                {0, 0.434601, 0.337407, 0.218091, 0, 0, 0, 0},
                                {0, 0, 0.218091, 0.452596, 0.381082, 0, 0, 0},
                                {0, 0, 0, 0.381082, 0.845901, 0.514452, 0, 0},
                                {0, 0, 0, 0, 0.514452, 0.392368, 0.065431, 0},
                                {0, 0, 0, 0, 0, 0.065431, 0.698868, 0.690582},
                                {0, 0, 0, 0, 0, 0, 0.690582, 0.606674}};
    int total_elements = 8 * 8;
    float *oneDArray = (float *)malloc(total_elements * sizeof(float));
    int index = 0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            oneDArray[index] = given_matrix[i][j];
            index++;
        }
    }
    Matrix result(8, oneDArray);
    free(oneDArray);
    return result;
}

// fixed matrix; matrixSize = 4, blocksize = 2, bandwidth = 1
// isSymmetric = true
Matrix generateFixedMatrixOfSize4() {
    int N = 4;
    float given_matrix[N][N] = {
        {1, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 3, 0}, {0, 0, 0, 4}};
    int total_elements = N * N;
    float *oneDArray = (float *)malloc(total_elements * sizeof(float));
    int index = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            oneDArray[index] = given_matrix[i][j];
            index++;
        }
    }
    Matrix result(N, oneDArray);
    free(oneDArray);
    return result;
}

/*
    Make a matrix symmetric;
*/
void transformToSymmetric(Matrix &A) {
    float *matrix = A.getMat();
    int matrixSize, blockSize;
    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = i; j < matrixSize; ++j) {
            matrix[j * matrixSize + i] = matrix[i * matrixSize + j];
        }
    }
}