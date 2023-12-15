#include "rgf1.hpp"
#include "matrices_utils.hpp"
#include <cassert>

// Print the matrix
void printMatrix(const float *matrix, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

void rgf1sided(Matrix &A, Matrix &G, bool sym_mat = false,
               bool save_off_diag = true) {
    int blockSize, matrixSize;
    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    // 0. Inverse of the first block
    A.invBLAS(blockSize, A.mdiag, G.mdiag); // I

    int nblocks = matrixSize / blockSize;

    // 1. Forward substitution (performed left to right)
    for (int i = 1; i < nblocks; ++i) {
        float *AAi = new float[blockSize * blockSize](),
              *AGi = new float[blockSize * blockSize]();
        A.mmmBLAS(blockSize, &(A.lodiag[(i - 1) * blockSize * blockSize]),
                  &(G.mdiag[(i - 1) * blockSize * blockSize]), AGi); // M
        A.mmmBLAS(blockSize, AGi, &(A.updiag[(i - 1) * blockSize * blockSize]),
                  AAi); // M
        A.mmSub(blockSize, &(A.mdiag[i * blockSize * blockSize]), AAi, AGi); // S
        A.invBLAS(blockSize, AGi, &(G.mdiag[i * blockSize * blockSize])); // I
    } // N / B - 1

    for (int i = nblocks - 2; i >= 0; --i) {
        float *Glf = new float[blockSize * blockSize](),
              *Glf1 = new float[blockSize * blockSize]();
        A.mmmBLAS(blockSize, &(G.mdiag[(i + 1) * blockSize * blockSize]),
                  &(A.lodiag[i * blockSize * blockSize]), Glf1); // M
        A.mmmBLAS(blockSize, Glf1, &(G.mdiag[i * blockSize * blockSize]), Glf); // M

        if (save_off_diag) {
            A.matScale(blockSize, Glf, -1,
                       &(G.lodiag[i * blockSize * blockSize])); // S
            if (sym_mat) {
                A.transposeBLAS(blockSize,
                                &(G.lodiag[i * blockSize * blockSize]),
                                &(G.updiag[i * blockSize * blockSize])); // S
            } else {
                float *Guf = new float[blockSize * blockSize](),
                      *Guf1 = new float[blockSize * blockSize]();
                A.mmmBLAS(blockSize, &(A.updiag[i * blockSize * blockSize]),
                          &(G.mdiag[(i + 1) * blockSize * blockSize]), Guf1); // M
                A.mmmBLAS(blockSize, &(G.mdiag[i * blockSize * blockSize]),
                          Guf1, Guf); // M
                A.matScale(blockSize, Guf, -1,
                           &(G.updiag[i * blockSize * blockSize])); // S

                delete[] Guf;
                delete[] Guf1;
            }
        } // N / B - 2

        A.mmmBLAS(blockSize, &(A.updiag[i * blockSize * blockSize]), Glf, Glf1); // M
        A.mmmBLAS(blockSize, &(G.mdiag[i * blockSize * blockSize]), Glf1, Glf); // M
        A.mmAdd(blockSize, &(G.mdiag[i * blockSize * blockSize]), Glf, 
                &(G.mdiag[i * blockSize * blockSize])); // S

        delete[] Glf;
        delete[] Glf1;
    }
}
