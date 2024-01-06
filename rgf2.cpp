#include "rgf2.hpp"

/**
 * @brief Performs a two-sided RGF inversion on a given matrix.
 *
 * This function performs a two-sided RGF inversion on a given matrix. The
 * inversion is performed in parallel using MPI, with different processes
 * handling different halves of the matrix.
 *
 * @param A The matrix on which the RGF inversion is to be performed.
 * @param G The matrix that will hold the result of the RGF inversion.
 * @param sym_mat A boolean flag indicating whether the input matrix is
 * symmetric.
 * @param save_off_diag A boolean flag indicating whether to save the
 * off-diagonal elements of the matrix.
 *
 * @return void
 *
 * @note This function assumes that the size of the matrix is divisible by the
 * block size, and that the number of blocks is divisible by 2.
 */
void rgf2sided(Matrix &A, Matrix &G, bool sym_mat, bool save_off_diag) {
    int processRank, blockSize, matrixSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    int nblocks = matrixSize / blockSize; // assume divisible
    int nblocks_2 = nblocks / 2;          // assume divisible

    if (processRank == 0) {
        rgf2sided_upperprocess(A, G, nblocks_2, sym_mat, save_off_diag);

        MPI_Recv((void *)(G.mdiag + nblocks_2 * blockSize * blockSize),
                 (nblocks_2)*blockSize * blockSize, MPI_FLOAT, 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Recv((void *)(G.updiag + nblocks_2 * blockSize * blockSize),
                 (nblocks_2 - 1) * blockSize * blockSize, MPI_FLOAT, 1, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Recv((void *)(G.lodiag + nblocks_2 * blockSize * blockSize),
                 (nblocks_2 - 1) * blockSize * blockSize, MPI_FLOAT, 1, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    } else if (processRank == 1) {
        rgf2sided_lowerprocess(A, G, nblocks - nblocks_2, sym_mat,
                               save_off_diag);
        MPI_Send((const void *)(G.mdiag + nblocks_2 * blockSize * blockSize),
                 (nblocks - nblocks_2) * blockSize * blockSize, MPI_FLOAT, 0, 0,
                 MPI_COMM_WORLD);

        MPI_Send((const void *)(G.updiag + nblocks_2 * blockSize * blockSize),
                 (nblocks - nblocks_2 - 1) * blockSize * blockSize, MPI_FLOAT,
                 0, 1, MPI_COMM_WORLD);

        MPI_Send((const void *)(G.lodiag + nblocks_2 * blockSize * blockSize),
                 (nblocks - nblocks_2 - 1) * blockSize * blockSize, MPI_FLOAT,
                 0, 2, MPI_COMM_WORLD);
    }
}

/**
 * @brief Performs the upper half of a two-sided RGF inversion on a given
 * matrix.
 *
 * This function performs the upper half of a two-sided RGF inversion on a given
 * matrix.
 *
 * @param A The matrix on which the RGF inversion is to be performed.
 * @param G The matrix that will hold the result of the RGF inversion.
 * @param nblocks_2 The number of blocks in the upper half of the matrix.
 * @param sym_mat A boolean flag indicating whether the input matrix is
 * symmetric.
 * @param save_off_diag A boolean flag indicating whether to save the
 * off-diagonal elements of the matrix.
 *
 * @return void
 *
 * @note This function assumes that the size of the matrix is divisible by the
 * block size, and that the number of blocks is divisible by 2.
 */
void rgf2sided_upperprocess(Matrix &A, Matrix &G, int nblocks_2, bool sym_mat,
                            bool save_off_diag) {
    int blockSize, matrixSize;
    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);

    int nblocks = matrixSize / blockSize;

    float *A_diagblk_leftprocess = A.mdiag;
    float *A_upperblk_leftprocess = A.updiag;
    float *A_lowerblk_leftprocess = A.lodiag;

    // Zero initialization
    float *G_diagblk_leftprocess =
        new float[(nblocks_2 + 1) * blockSize * blockSize]();
    float *G_upperblk_leftprocess =
        new float[nblocks_2 * blockSize * blockSize]();
    float *G_lowerblk_leftprocess =
        new float[nblocks_2 * blockSize * blockSize]();

    float *temp_result_1 = new float[blockSize * blockSize]();
    float *temp_result_2 = new float[blockSize * blockSize]();
    float *temp_result_3 = new float[blockSize * blockSize]();
    float *temp_result_4 = new float[blockSize * blockSize]();

    float *zeros = new float[blockSize * blockSize]();

    // Initialisation of g - invert first block
    A.invBLAS(blockSize, A_diagblk_leftprocess, G_diagblk_leftprocess);

    // Forward substitution
    for (int i = 1; i < nblocks_2; ++i) {
        A.mmmBLAS(blockSize,
                  A_lowerblk_leftprocess + (i - 1) * blockSize * blockSize,
                  G_diagblk_leftprocess + (i - 1) * blockSize * blockSize,
                  temp_result_1);

        A.mmmBLAS(blockSize, temp_result_1,
                  A_upperblk_leftprocess + (i - 1) * blockSize * blockSize,
                  temp_result_2);

        A.mmSub(blockSize, A_diagblk_leftprocess + i * blockSize * blockSize,
                temp_result_2, temp_result_2);

        A.invBLAS(blockSize, temp_result_2,
                  G_diagblk_leftprocess + i * blockSize * blockSize);
    }

    // Communicate the left connected block and receive the right connected
    // block
    MPI_Send((const void *)(G_diagblk_leftprocess +
                            (nblocks_2 - 1) * blockSize * blockSize),
             blockSize * blockSize, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);

    MPI_Recv(
        (void *)(G_diagblk_leftprocess + nblocks_2 * blockSize * blockSize),
        blockSize * blockSize, MPI_FLOAT, 1, 0, MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);

    // Connection from both sides of the full G
    A.mmmBLAS(blockSize,
              A_lowerblk_leftprocess + (nblocks_2 - 2) * blockSize * blockSize,
              G_diagblk_leftprocess + (nblocks_2 - 2) * blockSize * blockSize,
              temp_result_1);

    A.mmmBLAS(blockSize, temp_result_1,
              A_upperblk_leftprocess + (nblocks_2 - 2) * blockSize * blockSize,
              temp_result_2);

    A.mmmBLAS(blockSize,
              A_upperblk_leftprocess + (nblocks_2 - 1) * blockSize * blockSize,
              G_diagblk_leftprocess + (nblocks_2)*blockSize * blockSize,
              temp_result_3);

    A.mmmBLAS(blockSize, temp_result_3,
              A_lowerblk_leftprocess + (nblocks_2 - 1) * blockSize * blockSize,
              temp_result_4);

    A.mmSub(blockSize,
            A_diagblk_leftprocess + (nblocks_2 - 1) * blockSize * blockSize,
            temp_result_2, temp_result_2);

    A.mmSub(blockSize, temp_result_2, temp_result_4, temp_result_2);

    A.invBLAS(blockSize, temp_result_2,
              G_diagblk_leftprocess + (nblocks_2 - 1) * blockSize * blockSize);

    A.mmmBLAS(blockSize,
              G_diagblk_leftprocess + nblocks_2 * blockSize * blockSize,
              A_lowerblk_leftprocess + (nblocks_2 - 1) * blockSize * blockSize,
              temp_result_1);

    A.mmmBLAS(blockSize, temp_result_1,
              G_diagblk_leftprocess + (nblocks_2 - 1) * blockSize * blockSize,
              temp_result_2);

    A.mmSub(blockSize, zeros, temp_result_2,
            G_lowerblk_leftprocess + (nblocks_2 - 1) * blockSize * blockSize);

    if (sym_mat) {
        // matrix transpose
        A.transposeBLAS(
            blockSize,
            G_lowerblk_leftprocess + (nblocks_2 - 1) * blockSize * blockSize,
            G_upperblk_leftprocess + (nblocks_2 - 1) * blockSize * blockSize);
    } else {
        A.mmmBLAS(
            blockSize,
            G_diagblk_leftprocess + (nblocks_2 - 1) * blockSize * blockSize,
            A_upperblk_leftprocess + (nblocks_2 - 1) * blockSize * blockSize,
            temp_result_1);

        A.mmmBLAS(blockSize, temp_result_1,
                  G_diagblk_leftprocess + (nblocks_2)*blockSize * blockSize,
                  temp_result_2);

        A.mmSub(blockSize, zeros, temp_result_2,
                G_upperblk_leftprocess +
                    (nblocks_2 - 1) * blockSize * blockSize);
    }

    // Backward substitution
    for (int i = nblocks_2 - 2; i >= 0; i -= 1) {
        float *g_ii = G_diagblk_leftprocess + i * blockSize * blockSize;
        float *G_lowerfactor = new float[blockSize * blockSize];

        A.mmmBLAS(
            blockSize, G_diagblk_leftprocess + (i + 1) * blockSize * blockSize,
            A_lowerblk_leftprocess + i * blockSize * blockSize, temp_result_1);

        A.mmmBLAS(blockSize, temp_result_1, g_ii, G_lowerfactor);

        if (save_off_diag) {
            A.mmSub(blockSize, zeros, G_lowerfactor,
                    G_lowerblk_leftprocess + i * blockSize * blockSize);

            if (sym_mat) {
                // matrix transpose
                A.transposeBLAS(
                    blockSize,
                    G_lowerblk_leftprocess + (i)*blockSize * blockSize,
                    G_upperblk_leftprocess + (i)*blockSize * blockSize);
            } else {
                A.mmmBLAS(blockSize, g_ii,
                          A_upperblk_leftprocess + i * blockSize * blockSize,
                          temp_result_1);

                A.mmmBLAS(blockSize, temp_result_1,
                          G_diagblk_leftprocess +
                              (i + 1) * blockSize * blockSize,
                          temp_result_2);

                A.mmSub(blockSize, zeros, temp_result_2,
                        G_upperblk_leftprocess + i * blockSize * blockSize);
            }
        }

        A.mmmBLAS(blockSize, g_ii,
                  A_upperblk_leftprocess + i * blockSize * blockSize,
                  temp_result_1);

        A.mmmBLAS(blockSize, temp_result_1, G_lowerfactor, temp_result_2);

        A.mmAdd(blockSize, g_ii, temp_result_2,
                G_diagblk_leftprocess + i * blockSize * blockSize);

        delete[] G_lowerfactor;
    }

    memcpy(G.mdiag, G_diagblk_leftprocess,
           (nblocks_2 + 1) * blockSize * blockSize * sizeof(float));
    memcpy(G.updiag, G_upperblk_leftprocess,
           nblocks_2 * blockSize * blockSize * sizeof(float));
    memcpy(G.lodiag, G_lowerblk_leftprocess,
           nblocks_2 * blockSize * blockSize * sizeof(float));

    delete[] G_diagblk_leftprocess;
    delete[] G_upperblk_leftprocess;
    delete[] G_lowerblk_leftprocess;
    delete[] temp_result_1;
    delete[] temp_result_2;
    delete[] temp_result_3;
    delete[] temp_result_4;
    delete[] zeros;

    return;
}

/**
 * @brief Performs the lower half of a two-sided RGF inversion on a given
 * matrix.
 *
 * This function performs the lower half of a two-sided RGF inversion on a given
 * matrix.
 *
 * @param A The matrix on which the RGF inversion is to be performed.
 * @param G The matrix that will hold the result of the RGF inversion.
 * @param nblocks_2 The number of blocks in the lower half of the matrix.
 * @param sym_mat A boolean flag indicating whether the input matrix is
 * symmetric.
 * @param save_off_diag A boolean flag indicating whether to save the
 * off-diagonal elements of the matrix.
 *
 * @return void
 *
 * @note This function assumes that the size of the matrix is divisible by the
 * block size, and that the number of blocks is divisible by 2.
 */
void rgf2sided_lowerprocess(Matrix &A, Matrix &G, int nblocks_2, bool sym_mat,
                            bool save_off_diag) {
    int blockSize, matrixSize;
    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);

    float *A_diagblk_rightprocess = A.mdiag + nblocks_2 * blockSize * blockSize;
    float *A_upperblk_rightprocess =
        A.updiag + (nblocks_2 - 1) * blockSize * blockSize;
    float *A_lowerbk_rightprocess =
        A.lodiag + (nblocks_2 - 1) * blockSize * blockSize;
    float *G_diagblk_rightprocess =
        new float[(nblocks_2 + 1) * blockSize * blockSize]();
    float *G_upperblk_rightprocess =
        new float[nblocks_2 * blockSize * blockSize]();
    float *G_lowerblk_rightprocess =
        new float[nblocks_2 * blockSize * blockSize]();

    float *temp_result_1 = new float[blockSize * blockSize]();
    float *temp_result_2 = new float[blockSize * blockSize]();

    float *temp_result_3 = new float[blockSize * blockSize]();
    float *temp_result_4 = new float[blockSize * blockSize]();
    float *zeros = new float[blockSize * blockSize]();

    A.invBLAS(blockSize,
              A_diagblk_rightprocess + (nblocks_2 - 1) * blockSize * blockSize,
              G_diagblk_rightprocess + nblocks_2 * blockSize * blockSize);

    // Forward substitution
    for (int i = nblocks_2 - 1; i >= 1; i -= 1) {
        A.mmmBLAS(blockSize,
                  A_upperblk_rightprocess + i * blockSize * blockSize,
                  G_diagblk_rightprocess + (i + 1) * blockSize * blockSize,
                  temp_result_1);

        A.mmmBLAS(blockSize, temp_result_1,
                  A_lowerbk_rightprocess + i * blockSize * blockSize,
                  temp_result_2);

        A.mmSub(blockSize,
                A_diagblk_rightprocess + (i - 1) * blockSize * blockSize,
                temp_result_2, temp_result_2);

        A.invBLAS(blockSize, temp_result_2,
                  G_diagblk_rightprocess + i * blockSize * blockSize);
    }

    // Communicate the right connected block and receive the right connected
    // block

    MPI_Recv((void *)(G_diagblk_rightprocess), blockSize * blockSize, MPI_FLOAT,
             0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Send((const void *)(G_diagblk_rightprocess + 1 * blockSize * blockSize),
             blockSize * blockSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

    // Connection from both sides of the full G
    A.mmmBLAS(blockSize, A_lowerbk_rightprocess, G_diagblk_rightprocess,
              temp_result_1);

    A.mmmBLAS(blockSize, temp_result_1, A_upperblk_rightprocess, temp_result_2);

    A.mmmBLAS(blockSize, A_upperblk_rightprocess + (1) * blockSize * blockSize,
              G_diagblk_rightprocess + (2) * blockSize * blockSize,
              temp_result_3);

    A.mmmBLAS(blockSize, temp_result_3,
              A_lowerbk_rightprocess + (1) * blockSize * blockSize,
              temp_result_4);

    A.mmSub(blockSize, A_diagblk_rightprocess, temp_result_2, temp_result_2);

    A.mmSub(blockSize, temp_result_2, temp_result_4, temp_result_2);

    A.invBLAS(blockSize, temp_result_2,
              G_diagblk_rightprocess + (1) * blockSize * blockSize);

    // Compute the shared off-diagonal upper block
    A.mmmBLAS(blockSize, G_diagblk_rightprocess + 1 * blockSize * blockSize,
              A_lowerbk_rightprocess, temp_result_1);

    A.mmmBLAS(blockSize, temp_result_1, G_diagblk_rightprocess,
              G_lowerblk_rightprocess);

    if (sym_mat) {
        // matrix transpose
        A.transposeBLAS(blockSize, G_lowerblk_rightprocess,
                        G_upperblk_rightprocess);
    } else {
        A.mmmBLAS(blockSize, G_diagblk_rightprocess, A_upperblk_rightprocess,
                  temp_result_1);

        A.mmmBLAS(blockSize, temp_result_1,
                  G_diagblk_rightprocess + 1 * blockSize * blockSize,
                  G_upperblk_rightprocess);
    }

    // Backward substitution
    for (int i = 1; i < nblocks_2; ++i) {
        float *g_ii = G_diagblk_rightprocess + (i + 1) * blockSize * blockSize;
        float *G_lowerfactor = new float[blockSize * blockSize];

        A.mmmBLAS(blockSize, g_ii,
                  A_lowerbk_rightprocess + i * blockSize * blockSize,
                  temp_result_1);

        A.mmmBLAS(blockSize, temp_result_1,
                  G_diagblk_rightprocess + i * blockSize * blockSize,
                  G_lowerfactor);

        if (save_off_diag) {
            A.mmSub(blockSize, zeros, G_lowerfactor,
                    G_lowerblk_rightprocess + i * blockSize * blockSize);

            if (sym_mat) {
                // matrix transpose
                A.transposeBLAS(
                    blockSize,
                    G_lowerblk_rightprocess + i * blockSize * blockSize,
                    G_upperblk_rightprocess + i * blockSize * blockSize);
            } else {
                A.mmmBLAS(blockSize,
                          G_diagblk_rightprocess + i * blockSize * blockSize,
                          A_upperblk_rightprocess + i * blockSize * blockSize,
                          temp_result_1);

                A.mmmBLAS(blockSize, temp_result_1, g_ii, temp_result_2);

                A.mmSub(blockSize, zeros, temp_result_2,
                        G_upperblk_rightprocess + i * blockSize * blockSize);
            }
        }

        A.mmmBLAS(blockSize, G_lowerfactor,
                  A_upperblk_rightprocess + i * blockSize * blockSize,
                  temp_result_1);

        A.mmmBLAS(blockSize, temp_result_1, g_ii, temp_result_2);

        A.mmAdd(blockSize, g_ii, temp_result_2,
                G_diagblk_rightprocess + (i + 1) * blockSize * blockSize);
    }

    memcpy(G.mdiag + nblocks_2 * blockSize * blockSize,
           G_diagblk_rightprocess + blockSize * blockSize,
           nblocks_2 * blockSize * blockSize * sizeof(float));
    memcpy(G.updiag + (nblocks_2 - 1) * blockSize * blockSize,
           G_upperblk_rightprocess,
           nblocks_2 * blockSize * blockSize * sizeof(float));
    memcpy(G.lodiag + (nblocks_2 - 1) * blockSize * blockSize,
           G_lowerblk_rightprocess,
           nblocks_2 * blockSize * blockSize * sizeof(float));

    delete[] G_diagblk_rightprocess;
    delete[] G_upperblk_rightprocess;
    delete[] G_lowerblk_rightprocess;
    delete[] temp_result_1;
    delete[] temp_result_2;
    delete[] zeros;
    delete[] temp_result_3;
    delete[] temp_result_4;

    return;
}

// #include "argparse.h"
// #include "rgf1.hpp"
// typedef struct {
//     int matrixSize;
//     int blockSize;
//     int numRuns;
//     bool isSymmetric;
//     bool saveOffDiag;
//     char *inputPath;
// } Config;

// void InitOptions(Config *config) {
//     config->blockSize = 2;
//     config->matrixSize = 0;
//     config->numRuns = 10;
//     config->isSymmetric = false;
//     config->saveOffDiag = true;
//     config->inputPath = NULL;
// }

// int parse(Config *config, int argc, const char **argv) {
//     static const char *const usages[] = {
//         NULL,
//     };
//     struct argparse_option options[] = {
//         OPT_HELP(),
//         OPT_INTEGER('m', "matrixSize", &config->matrixSize, "matrix size",
//         NULL,
//                     0, 0),
//         OPT_INTEGER('b', "blockSize", &config->blockSize, "block size", NULL,
//         0,
//                     0),
//         OPT_INTEGER('n', "numRuns", &config->numRuns, "number of runs", NULL,
//         0,
//                     0),
//         OPT_INTEGER('s', "isSymmetric", &config->isSymmetric, "is symmetric",
//                     NULL, 0, 0),
//         OPT_INTEGER('o', "saveOffDiag", &config->saveOffDiag, "save off
//         diag", NULL, 0, 0), OPT_STRING('f', "inputPath", &config->inputPath,
//         "input path", NULL, 0, 0), OPT_END(),
//     };

//     struct argparse argparse;
//     argparse_init(&argparse, options, usages, 0);
//     argparse_describe(&argparse, "DPHPC TEAM", NULL);
//     argc = argparse_parse(&argparse, argc, argv);

//     return 0;
// }

// // TEMP main to test stuff out
// int main(int argc, const char *argv[]) {
//     int processRank;
//     const char *bin_name = argv[0];
//     Config config;
//     InitOptions(&config);
//     parse(&config, argc, argv);

//     MPI_Init(&argc, (char ***)(&argv));
//     MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

//     if (config.inputPath != NULL) {
//         // read matrix from file
//     } else if (config.matrixSize != 0) {
//         // generate matrix
//         int MATRIX_SIZE = config.matrixSize;
//         int BLOCK_SIZE = config.blockSize;
//         assert(MATRIX_SIZE % BLOCK_SIZE == 0);
//         int NUM_RUNS = config.numRuns;
//         bool IS_SYMMETRIC = config.isSymmetric;
//         bool SAVE_OFF_DIAG = config.saveOffDiag;

//         Matrix inputMatrix =
//             generateBandedDiagonalMatrix(MATRIX_SIZE, BLOCK_SIZE,
//             IS_SYMMETRIC, 0);

//         Matrix tempResult(
//             MATRIX_SIZE); // zero initialization, same shape as inputMatrix
//         tempResult.convertDenseToBlkTridiag(
//             BLOCK_SIZE); // G has same blockSize as inputMatrix
//         rgf2sided(inputMatrix, tempResult, IS_SYMMETRIC, SAVE_OFF_DIAG);

//         if (processRank == 0) {
//             std::cout << "\n\nrgf2 RESULT\n\n";
//             tempResult.printB();
//         }

//         // compare agains rgf1
//         Matrix tempResult_cpp(MATRIX_SIZE); // zero initialization, same
//         shape as inputMatrix
//         tempResult_cpp.convertDenseToBlkTridiag(BLOCK_SIZE); // G has same
//         blockSize as inputMatrix

//         rgf1sided(inputMatrix, tempResult, IS_SYMMETRIC, SAVE_OFF_DIAG);
//         if (processRank == 0) {
//             std::cout << "\n\nrgf2 RESULT\n\n";
//             tempResult.printB();
//         }

//     }
//     MPI_Finalize();
// }