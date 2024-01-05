#include "matrices_utils.hpp"
// memory safety -- take care

Matrix::~Matrix() {
    delete[] mdiag;
    delete[] updiag;
    delete[] lodiag;
}

void Matrix::copyMatrixData(const Matrix &other) {
    // deep copy
    matrixSize = other.matrixSize;
    blockSize = other.blockSize;
    int nblocks = matrixSize / blockSize;
    mdiag = new float[nblocks * blockSize * blockSize];
    updiag = new float[(nblocks - 1) * blockSize * blockSize];
    lodiag = new float[(nblocks - 1) * blockSize * blockSize];
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
    blockSize = 1;
    mdiag = NULL;
    updiag = NULL;
    lodiag = NULL;
}

/* initlize matrix with size N and values newMat */
Matrix::Matrix(int N, float *newMat) {
    matrixSize = N;
    blockSize = 1;
    mdiag = NULL;
    updiag = NULL;
    lodiag = NULL;
}

void Matrix::transposeBLAS(int n, float *A, float *result) {
    cblas_somatcopy(CblasRowMajor, CblasTrans, n, n, 1.0f, A, n, result, n);
}

/* Generate 3 representations.
   Note: this function has not the same behavior as the name.
   It allocates the diagonal, upper, and lower blocks. 
   The content is not initialized, which is filled in generateBandedDiagonalMatrix
*/
void Matrix::convertDenseToBlkTridiag(const int blockSize) {
    this->blockSize = blockSize;
    assert(matrixSize % blockSize ==
           0); // matrixSize must be divisible by blockSize
    int nblocks = matrixSize / blockSize;
    mdiag = new float[nblocks * blockSize * blockSize];
    updiag = new float[(nblocks - 1) * blockSize * blockSize];
    lodiag = new float[(nblocks - 1) * blockSize * blockSize];
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

void Matrix::mmmBLAS(int n, float *A, float *B, float *result) {
    const CBLAS_ORDER order = CblasRowMajor;
    const CBLAS_TRANSPOSE transA = CblasNoTrans;
    const CBLAS_TRANSPOSE transB = CblasNoTrans;
    const float alpha = 1.0;
    const float beta = 0.0;

    cblas_sgemm(order, transA, transB, n, n, n, alpha, A, n, B, n, beta, result,
                n);
}

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

void Matrix::invBLAS(int n, const float *A, float *result) {

    int *ipiv = (int *)malloc(n * sizeof(int));
    memcpy(result, A, n * n * sizeof(float));

    LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, result, n, ipiv);

    LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, result, n, ipiv);

    free(ipiv);
}

void Matrix::matScale(int n, float *A, int k, float *result) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i * n + j] = A[i * n + j] * k;
        }
    }
}

void generateRandomMat(int matrixSize, bool isSymmetric, float *matrix) {
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
}

/* Note: this function has not the same behavior as the name.
   It allocate only the diagonal, upper, and lower blocks. 
   The matrix function has no internal representation. 
   This is done for fast calculation. In real scenario, this is not desired.
*/
Matrix generateBandedDiagonalMatrix(int matrixSize, int blockSize,
                                    bool isSymmetric, int seed) {
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
        generateRandomMat(blockSize, isSymmetric,
                          A.mdiag + i * blockSize * blockSize);
    }

    for (int i = 0; i < numBlocks - 1; ++i) {
        generateRandomMat(blockSize, false,
                          A.updiag + i * blockSize * blockSize);
        if (isSymmetric) {
            for (int k = 0; k < blockSize; ++k) {
                for (int j = 0; j < blockSize; ++j) {
                    A.lodiag[i * blockSize * blockSize + k * blockSize + j] =
                        A.updiag[i * blockSize * blockSize + j * blockSize + k];
                }
            }
        } else {
            generateRandomMat(blockSize, false,
                              A.lodiag + i * blockSize * blockSize);
        }
    }
    return A;
}
