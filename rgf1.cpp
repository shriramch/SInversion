#include "rgf1.hpp"
#include "matrices_utils.hpp"
#include <cassert>

void rgf1sided(Matrix &A, Matrix &G, bool sym_mat = false,
               bool save_off_diag = true) {
    int blockSize, matrixSize;
    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    // 0. Inverse of the first block
    A.invBLAS(blockSize, A.mdiag, G.mdiag);

    int nblocks = matrixSize / blockSize;

    // 1. Forward substitution (performed left to right)
    for (int i = 1; i < nblocks; ++i) {
        float *AAi = new float[blockSize * blockSize](),
              *AGi = new float[blockSize * blockSize]();
        A.mmmBLAS(blockSize, &(A.lodiag[(i - 1) * blockSize * blockSize]),
                  &(G.mdiag[(i - 1) * blockSize * blockSize]), AGi);
        A.mmmBLAS(blockSize, AGi, &(A.updiag[(i - 1) * blockSize * blockSize]),
                  AAi);
        A.mmSub(blockSize, &(A.mdiag[i * blockSize * blockSize]), AAi, AGi);
        A.invBLAS(blockSize, AGi, &(G.mdiag[i * blockSize * blockSize]));

        delete[] AAi;
        delete[] AGi;
    }

    for (int i = nblocks - 2; i >= 0; --i) {
        float *Glf = new float[blockSize * blockSize](),
              *Glf1 = new float[blockSize * blockSize]();
        A.mmmBLAS(blockSize, &(G.mdiag[(i + 1) * blockSize * blockSize]),
                  &(A.lodiag[i * blockSize * blockSize]), Glf1);
        A.mmmBLAS(blockSize, Glf1, &(G.mdiag[i * blockSize * blockSize]), Glf);

        if (save_off_diag) {
            A.matScale(blockSize, Glf, -1,
                       &(G.lodiag[i * blockSize * blockSize]));
            if (sym_mat) {
                A.transposeBLAS(blockSize,
                                &(G.lodiag[i * blockSize * blockSize]),
                                &(G.updiag[i * blockSize * blockSize]));
            } else {
                float *Guf = new float[blockSize * blockSize](),
                      *Guf1 = new float[blockSize * blockSize]();
                A.mmmBLAS(blockSize, &(A.updiag[i * blockSize * blockSize]),
                          &(G.mdiag[(i + 1) * blockSize * blockSize]), Guf1);
                A.mmmBLAS(blockSize, &(G.mdiag[i * blockSize * blockSize]),
                          Guf1, Guf);
                A.matScale(blockSize, Guf, -1,
                           &(G.updiag[i * blockSize * blockSize]));

                delete[] Guf;
                delete[] Guf1;
            }
        }

        A.mmmBLAS(blockSize, &(A.updiag[i * blockSize * blockSize]), Glf, Glf1);
        A.mmmBLAS(blockSize, &(G.mdiag[i * blockSize * blockSize]), Glf1, Glf);
        A.mmAdd(blockSize, &(G.mdiag[i * blockSize * blockSize]), Glf,
                &(G.mdiag[i * blockSize * blockSize]));

        delete[] Glf;
        delete[] Glf1;
    }
}
