#include "matrices_utils.hpp"
#include <cmath>
#include <assert.h>

bool areFloatsEqual(float a, float b, float epsilon = 1e-9) {
    return std::fabs(a - b) < epsilon;
}

void test_rgf2sided(int matrixSize, int blockSize, bool isSymmetric=false) {
    int processRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    int bandwidth = (int) std::ceilf(((float) blockSize) / 2); // consider some examples where blockSize is not even
    Matrix A =  generateBandedDiagonalMatrix(matrixSize, bandwidth, isSymmetric);
    A.convertDenseToBlkTridiag(blockSize); // calculate the diagonal blocks
    Matrix G(matrixSize); // zero initialization, same shape as A
    G.convertDenseToBlkTridiag(blockSize); // allocate memory for diagonal blocks
    rgf2sided(A, G, false, false);

    G.printM();
    G.printB();
    float *A_ref = new float[matrixSize * matrixSize];
    A.invBLAS(matrixSize, A.getMat(), A_ref);
    float *G_mat = G.getMat();
    
    if (processRank == 0) {
        for (int i = 0; i < matrixSize*matrixSize; ++i) {
        assert(areFloatsEqual(A_ref[i], G_mat[i]));
    }
        std::cout << "Test passed.\n";
    }
} 

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int processRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    if (processRank == 0) {
        std::cout << "Testing rgf2sided...\n";
        test_rgf2sided(16,2,true);
    }
    MPI_Finalize();
}