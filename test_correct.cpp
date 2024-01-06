#include "rgf2_cuda.hpp"
#include "argparse.h"
#include "rgf2.hpp"
#include "rgf1.hpp"
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <mpi.h>
/*
  general workflow to print the result of each implemenation. No need to compile each main under the implementation. 
  we can compare them manually. If all of them are same, we have some certaninty that the result is correct.
  Didn't compare with the blas inversion (this is done before) since we remove the matrix interval representation.
*/

int processRank;
typedef struct {
    int matrixSize;
    int blockSize;
    int numRuns;
    bool isSymmetric;
    bool saveOffDiag;
    char *inputPath;
} Config;

void InitOptions(Config *config) {
    config->blockSize = 2;
    config->matrixSize = 0;
    config->numRuns = 10;
    config->isSymmetric = false;
    config->saveOffDiag = true;
    config->inputPath = NULL;
}

int parse(Config *config, int argc, const char **argv) {
    static const char *const usages[] = {
        NULL,
    };
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_INTEGER('m', "matrixSize", &config->matrixSize, "matrix size",
        NULL,
                    0, 0),
        OPT_INTEGER('b', "blockSize", &config->blockSize, "block size", NULL,
        0,
                    0),
        OPT_INTEGER('n', "numRuns", &config->numRuns, "number of runs", NULL,
        0,
                    0),
        OPT_INTEGER('s', "isSymmetric", &config->isSymmetric, "is symmetric",
                    NULL, 0, 0),
        OPT_INTEGER('o', "saveOffDiag", &config->saveOffDiag, "save off diag", NULL, 0, 0),
        OPT_STRING('f', "inputPath", &config->inputPath, "input path", NULL, 0, 0),
        OPT_END(),
    };

    struct argparse argparse;
    argparse_init(&argparse, options, usages, 0);
    argparse_describe(&argparse, "DPHPC TEAM", NULL);
    argc = argparse_parse(&argparse, argc, argv);

    return 0;
}

// TEMP main to test stuff out
int main(int argc, const char *argv[]) {
    const char *bin_name = argv[0];
    Config config;
    InitOptions(&config);
    parse(&config, argc, argv);

    MPI_Init(&argc, (char ***)(&argv));
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    if (config.inputPath != NULL) {
        // read matrix from file
    } else if (config.matrixSize != 0) {
        // generate matrix
        int MATRIX_SIZE = config.matrixSize;
        int BLOCK_SIZE = config.blockSize;
        assert(MATRIX_SIZE % BLOCK_SIZE == 0);
        int NUM_RUNS = config.numRuns;
        bool IS_SYMMETRIC = config.isSymmetric;
        bool SAVE_OFF_DIAG = config.saveOffDiag;

        Matrix inputMatrix =
            generateBandedDiagonalMatrix(MATRIX_SIZE, BLOCK_SIZE, IS_SYMMETRIC, 0);
        
        // if(processRank == 0) {
        //     inputMatrix.printB();
        // }
        // MPI_Barrier(MPI_COMM_WORLD);
        // if(processRank == 1) {
        //     inputMatrix.printB();
        // }

        Matrix tempResult_rgf1(MATRIX_SIZE); // zero initialization, same shape as inputMatrix
        tempResult_rgf1.convertDenseToBlkTridiag(BLOCK_SIZE); // G has same blockSize as inputMatrix
        // Check against the already implemented RGF1 on C++
        rgf1sided(inputMatrix, tempResult_rgf1, IS_SYMMETRIC, SAVE_OFF_DIAG);

        Matrix tempResult_rgf2(MATRIX_SIZE); // zero initialization, same shape as inputMatrix
        tempResult_rgf2.convertDenseToBlkTridiag(BLOCK_SIZE); // G has same blockSize as inputMatrix
        rgf2sided(inputMatrix, tempResult_rgf2, IS_SYMMETRIC, SAVE_OFF_DIAG);

        Matrix tempResult_rgf2_cuda(MATRIX_SIZE); // zero initialization, same shape as inputMatrix
        tempResult_rgf2_cuda.convertDenseToBlkTridiag(BLOCK_SIZE); // G has same blockSize as inputMatrix
        rgf2sided_cuda(inputMatrix, tempResult_rgf2_cuda, IS_SYMMETRIC, SAVE_OFF_DIAG);

        
        if (processRank == 0) {
            // if (!tempResult_rgf1.compareDiagonals(tempResult_rgf2, false)) {
            //     std::cout << "Error while running rgf2:" << std::endl;
            //     return -1;
            // }
            // printf("rgf2 algorithm test passed!\n"); 
            if (!tempResult_rgf2.compareDiagonals(tempResult_rgf2_cuda, false)) {
                std::cout << "Error while running rgf2_cuda:" << std::endl;
                return -1;
            }
            printf("rgf2_cuda algorithm test passed!\n"); 
        }


        // if (processRank == 0) {
        //     std::cout << "\n\nrgf1 RESULT \n\n";
        //     tempResult_rgf1.printB();
        // }

        // if (processRank == 0) {
        //     std::cout << "\n\nrgf2 RESULT \n\n";
        //     tempResult_rgf2.printB();
        // }

        // if (processRank == 0) {
        //     std::cout << "\n\nrgf2_CUDA RESULT\n\n";
        //     tempResult_rgf2_cuda.printB();
        // }

    }
    MPI_Finalize();
}