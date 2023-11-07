#include "rgf2.hpp"

generateBandedDiagonalMatrix() {

    
    A = generateRandomNumpyMat(matrice_size, is_complex, is_symmetric, seed)
    
    for i in range(matrice_size):
        for j in range(matrice_size):
            if i - j > matrice_bandwidth or j - i > matrice_bandwidth:
                A[i, j] = 0

    return A
}

convertDenseToBlkTridiag

test_rgf2sided() {

}

int main(int argc, char **argv) {
    int processRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    float *t = new float[16];
    for (int i = 0; i < 16; ++i) {
        t[i] = i + 1;
    }
    Matrix A(4, t);
    A.printM();
    A.convert3D(2);
    A.printB();

    Matrix G(4); // zero initialization, same shape as A
    G.convert3D(2); // G has same blockSize as in A
    // rgf2sided(A, G,false, false);

    MPI_Finalize();
}