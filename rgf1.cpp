#include "rgf1.hpp"
#include "matrices_utils.hpp"
#include <cassert>
#include <mpi.h>

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

// #include "argparse.h"
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
//         OPT_INTEGER('o', "saveOffDiag", &config->saveOffDiag, "save off diag", NULL, 0, 0),
//         OPT_STRING('f', "inputPath", &config->inputPath, "input path", NULL, 0, 0),
//         OPT_END(),
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
//             generateBandedDiagonalMatrix(MATRIX_SIZE, BLOCK_SIZE, IS_SYMMETRIC, 0);

//         Matrix tempResult(
//             MATRIX_SIZE); // zero initialization, same shape as inputMatrix
//         tempResult.convertDenseToBlkTridiag(
//             BLOCK_SIZE); // G has same blockSize as inputMatrix
//         rgf1sided(inputMatrix, tempResult, IS_SYMMETRIC, SAVE_OFF_DIAG);

//         // no one to compare, just print. Correctness is checked
//         if (processRank == 0) {
//             std::cout << "\n\nrgf1 RESULT\n\n";
//             tempResult.printB();
//         }    
//     }
//     MPI_Finalize();
// }
