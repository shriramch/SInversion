#include "matrices_utils.hpp"
// memory safety -- take care
/*
    Generate a dense matrix of shape: (matrice_size x matrice_size) filled 
    with random numbers. The matrice should real valued.
*/
Matrix generateRandomMat(int matrixSize, bool isSymmetric, int seed) {
    Matrix mat(matrixSize);
    float *matrix = mat.getMat();
    if (seed != 0) {
        // Seed the random number generator if a seed is provided
        std::srand(seed);
    } else {
        // Seed the random number generator with the current time if no seed is provided
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
    }

    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            // Generate a random real-valued number between 0 and 1
            float random_number = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

            // Store the number in the matrix
            matrix[i * matrixSize + j] = random_number;

            if (isSymmetric && i != j) {
                // If the matrix is symmetric, set the corresponding off-diagonal element
                matrix[j * matrixSize + i] = random_number;
            }
        }
    }
    return mat;
}

/* 
    Generate a banded diagonal matrix of shape: matrice_size^2 with a 
    bandwidth = matrice_bandwidth, filled with random numbers.
*/
Matrix generateBandedDiagonalMatrix(int matrixSize, int matriceBandwidth, bool isSymmetric, int seed) {
    Matrix A = generateRandomMat(matrixSize, isSymmetric, seed);
    float* matrix = A.getMat();
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            if (i - j > matriceBandwidth || j - i > matriceBandwidth) {
                matrix[i * matrixSize+j] = 0;
            }
        }
    }
    return A;
}

Matrix generateFixedMatrixOfSize8 () {
    float given_matrix[8][8] = {
        {0.11214, 0.976047, 0, 0, 0, 0, 0, 0},
        {0.976047, 0.423681, 0.434601, 0, 0, 0, 0, 0},
        {0, 0.434601, 0.337407, 0.218091, 0, 0, 0, 0},
        {0, 0, 0.218091, 0.452596, 0.381082, 0, 0, 0},
        {0, 0, 0, 0.381082, 0.845901, 0.514452, 0, 0},
        {0, 0, 0, 0, 0.514452, 0.392368, 0.065431, 0},
        {0, 0, 0, 0, 0, 0.065431, 0.698868, 0.690582},
        {0, 0, 0, 0, 0, 0, 0.690582, 0.606674}
    };
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

/* 
    Make a matrix symmetric;
*/
void transformToSymmetric(Matrix& A) {
    float* matrix = A.getMat();
    int matrixSize, blockSize;
    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = i; j < matrixSize; ++j) {
            matrix[j * matrixSize + i] = matrix[i * matrixSize + j];
        }
    }
}


// test function correctness
// int main() {
//     int matrice_size = 8;
//     Matrix m = generateBandedDiagonalMatrix(matrice_size, true, 0);
//     float *matrix = m.getMat(); //generateRandomMat(matrice_size, true, 0);

//     for (int i = 0; i < matrice_size; i++) {
//         for (int j = 0; j < matrice_size; j++) {
//             std::cout << matrix[i * matrice_size + j] << " ";
//         }
//         std::cout << std::endl;
//     }
//     return 0;
// }
