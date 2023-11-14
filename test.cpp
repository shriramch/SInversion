#include "argparse.h"
#include "matrices_utils.hpp"
#include "rgf2.hpp"
#include <stdio.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
// Function pointer type for the algorithm (so is easier to test them, they
// would all expect a Matrix and return another one)
typedef std::function<void(Matrix &, Matrix &, int, int, bool, bool)>
    AlgorithmFunction; // input as originalMatrix, resultMatrix, MATRIX_SIZE,
                       // BLOCK_SIZE

// Base RGF 2-Sided algorithm
void rgf2sidedAlgorithm(Matrix &input, Matrix &result, int matrixSize,
                        int blockSize, bool is_symmetric = false,
                        bool save_off_diag = true) {
    rgf2sided(input, result, is_symmetric, save_off_diag);
}

// Another algorithm to test
// void anotherAlgorithmconst Matrix& input, Matrix& result, int matrixSize, int
// blockSize) {
//     // Implement your other algorithm here
//     // ...
// }

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
    int processRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_INTEGER('m', "matrixSize", &config->matrixSize, "matrix size", NULL,
                    0, 0),
        OPT_INTEGER('b', "blockSize", &config->blockSize, "block size", NULL, 0,
                    0),
        OPT_INTEGER('n', "numRuns", &config->numRuns, "number of runs", NULL, 0,
                    0),
        OPT_INTEGER('s', "isSymmetric", &config->isSymmetric, "is symmetric",
                    NULL, 0, 0),
        OPT_INTEGER('o', "saveOffDiag", &config->saveOffDiag, "save off diag",
                    NULL, 0, 0),
        OPT_STRING('f', "inputPath", &config->inputPath, "input path", NULL, 0,
                   0),
        OPT_END(),
    };

    struct argparse argparse;
    argparse_init(&argparse, options, usages, 0);
    argparse_describe(&argparse, "DPHPC TEAM", NULL);
    argc = argparse_parse(&argparse, argc, argv);
    if (processRank == 0) {
        printf("[Config] matrixSize: %d\n", config->matrixSize);
        printf("[Config] blockSize: %d\n", config->blockSize);
        printf("[Config] numRuns: %d\n", config->numRuns);
        printf("[Config] isSymmetric: %d\n", config->isSymmetric);
        printf("[Config] saveOffDiag: %d\n", config->saveOffDiag);
        if (config->inputPath != NULL) {
            printf("[Config] inputPath: %s\n", config->inputPath);
        }
    }
    return 0;
}

int main(int argc, const char *argv[]) {
    const char *bin_name = argv[0];
    int processRank;
    MPI_Init(&argc, (char ***)(&argv));
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    // parse options
    Config config;
    InitOptions(&config);
    parse(&config, argc, argv);
    if (config.inputPath != NULL) {
        // read matrix from file
    } else if (config.matrixSize != 0) {
        // generate matrix
        // Note, skipping the whole MPI thing here as it will be done internally
        // in each function if needed
        int MATRIX_SIZE = config.matrixSize;
        int BLOCK_SIZE =
            config.blockSize; // MATRIX_SIZE should be divisible by this
        int NUM_RUNS = config.numRuns;
        bool IS_SYMMETRIC = config.isSymmetric;
        bool SAVE_OFF_DIAG = config.saveOffDiag;

        // Vector of algorithm functions and their names
        std::vector<std::pair<AlgorithmFunction, std::string>> algorithms = {
            {rgf2sidedAlgorithm, "rgf2sidedAlgorithm"}
            // {anotherAlgorithm, "Another Algorithm"}
            // Add more algorithms as needed
        };

        Matrix inputMatrix =
            generateBandedDiagonalMatrix(MATRIX_SIZE, 2, true, 0);
        // Matrix inputMatrix = generateFixedMatrixOfSize8();
        inputMatrix.convertDenseToBlkTridiag(BLOCK_SIZE);

        // use blas inv result as base result
        float *base_inv = new float[MATRIX_SIZE * MATRIX_SIZE]();
        inputMatrix.invBLAS(MATRIX_SIZE, inputMatrix.getMat(), base_inv);
        Matrix baseResultMatrix(MATRIX_SIZE, base_inv);
        baseResultMatrix.convertDenseToBlkTridiag(config.blockSize);

        // precision is low; the larger the matrix , the lower the precision
        // compare it to the blas inv result to test the correctness
        for (const auto &algorithm : algorithms) {
            Matrix tempResult(
                MATRIX_SIZE); // zero initialization, same shape as inputMatrix
            tempResult.convertDenseToBlkTridiag(
                BLOCK_SIZE); // G has same blockSize as inputMatrix
            algorithm.first(inputMatrix, tempResult, MATRIX_SIZE, BLOCK_SIZE,
                            IS_SYMMETRIC, SAVE_OFF_DIAG);
            if (processRank == 0) {
                if (!baseResultMatrix.compareDiagonals(tempResult, false)) {
                    std::cout << "Error while running the function:"
                              << algorithm.second << std::endl;
                    return -1;
                }
                std::cout << "test passed!\n";
            }
        }
        // Run all the functions for x times and measure the time
        // create just a single output, as i have already checked correctness,
        // now just need the timing of multiple runs

        Matrix tempResult(
            MATRIX_SIZE); // zero initialization, same shape as inputMatrix
        tempResult.convertDenseToBlkTridiag(
            4); // G has same blockSize as in inputMatrix
        for (const auto &algorithm : algorithms) {
            for (int i = 0; i < NUM_RUNS; ++i) {

                auto start =
                    clock(); // decide whether to use this or just clock()
                // Run the algorithm
                algorithm.first(inputMatrix, tempResult, MATRIX_SIZE,
                                BLOCK_SIZE, IS_SYMMETRIC, SAVE_OFF_DIAG);
                auto end = clock();
                auto duration = MAX(1, (end - start));
                // Output the time taken for each function
                if (processRank == 0) {
                    // write to file or accumulate
                    // std::cout << algorithm.second << " Time: " << duration <<
                    // std::endl;
                }
            }
        }
    } else if (processRank == 0) {
        printf("Usage (random mode): mpirun -np 2 %s -m <matrixSize> -b "
               "<blockSize=2> -n <numRuns=10> -s <isSymmetric=false> -o "
               "<saveOffDiag=true>\n",
               bin_name);
        printf("or (file mode): mpirun -np 2 %s -m <matrixSize> -b "
               "<blockSize=2> -n <numRuns=10> -s <isSymmetric=false> -o "
               "<saveOffDiag=true> -f <inputPath>\n",
               bin_name);
        return 1;
    }
    MPI_Finalize();
    return 0;
}
