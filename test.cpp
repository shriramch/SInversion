#include "argparse.h"
#include "matrices_utils.hpp"
#include "rgf1.hpp"
#include "rgf1_cuda.hpp"
#include "rgf2_cuda.hpp"
#include "rgf2.hpp"

#if defined ENABLE_LIBLSB1 || defined ENABLE_LIBLSB2 || defined ENABLE_LIBLSB_C1 || defined ENABLE_LIBLSB_C2
#include "liblsb.h"
#endif

#include <cmath>
#include <functional>
#include <iostream>
#include <stdio.h>
#include <vector>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

// Function pointer type for the algorithm
typedef std::function<void(Matrix &, Matrix &, int, int, bool, bool)>
    AlgorithmFunction;

#if defined ENABLE_LIBLSB2
void rgf2sidedAlgorithm(Matrix &input, Matrix &result, int matrixSize,
                        int blockSize, bool is_symmetric = false,
                        bool save_off_diag = true) {
    rgf2sided(input, result, is_symmetric, save_off_diag);
}
#endif
#if defined ENABLE_LIBLSB1
void rgf1sidedAlgorithm(Matrix &input, Matrix &result, int matrixSize,
                        int blockSize, bool is_symmetric = false,
                        bool save_off_diag = true) {
    rgf1sided(input, result, is_symmetric, save_off_diag);
}
#endif

#if defined ENABLE_LIBLSB_C1
void rgf1sidedCUDAAlgorithm(Matrix &input, Matrix &result, int matrixSize,
                            int blockSize, bool is_symmetric = false,
                            bool save_off_diag = true) {
    rgf1sided_cuda(input, result, is_symmetric, save_off_diag);
}
#endif

#if defined ENABLE_LIBLSB_C2
void rgf2sidedCUDAAlgorithm(Matrix &input, Matrix &result, int matrixSize,
                            int blockSize, bool is_symmetric = false,
                            bool save_off_diag = true) {
    rgf2sided_cuda(input, result, is_symmetric, save_off_diag);
}
#endif

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

#if defined ENABLE_LIBLSB1 || defined ENABLE_LIBLSB2 || defined ENABLE_LIBLSB_C1 || defined ENABLE_LIBLSB_C2
    LSB_Init("DPHPC_Project", 0);
#endif

    Config config;
    InitOptions(&config);
    parse(&config, argc, argv);
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

        // Vector of algorithm functions and their names
        std::vector<std::pair<AlgorithmFunction, std::string>> algorithms = {
#if defined ENABLE_LIBLSB1
            {rgf1sidedAlgorithm, "rgf1sidedAlgorithm"}
#elif defined ENABLE_LIBLSB2
            {rgf2sidedAlgorithm, "rgf2sidedAlgorithm"}
#elif defined ENABLE_LIBLSB_C1
            {rgf1sidedCUDAAlgorithm, "rgf1CUDA"}
#elif defined ENABLE_LIBLSB_C2
            {rgf2sidedCUDAAlgorithm, "rgf2CUDA"}
#else
            {rgf1sidedAlgorithm, "rgf1sidedAlgorithm"},
            {rgf2sidedAlgorithm, "rgf2sidedAlgorithm"},
        // {rgf1sidedCUDAAlgorithm, "rgf1sidedCUDAAlgorithm"},
#endif
        };

        Matrix inputMatrix =
            generateBandedDiagonalMatrix(MATRIX_SIZE, 2, true, 0);
        inputMatrix.convertDenseToBlkTridiag(BLOCK_SIZE);

        float *base_inv = new float[MATRIX_SIZE * MATRIX_SIZE]();
        inputMatrix.invBLAS(MATRIX_SIZE, inputMatrix.getMat(), base_inv);
        Matrix baseResultMatrix(MATRIX_SIZE, base_inv);
        baseResultMatrix.convertDenseToBlkTridiag(BLOCK_SIZE);
#if !defined ENABLE_LIBLSB1 && !defined ENABLE_LIBLSB2 && !defined ENABLE_LIBLSB_C1 && !defined ENABLE_LIBLSB_C2
        for (const auto &algorithm : algorithms) {
            Matrix tempResult(MATRIX_SIZE);
            tempResult.convertDenseToBlkTridiag(BLOCK_SIZE);
            algorithm.first(inputMatrix, tempResult, MATRIX_SIZE, BLOCK_SIZE,
                            IS_SYMMETRIC, SAVE_OFF_DIAG);
            if (processRank == 0) {
                if (!baseResultMatrix.compareDiagonals(tempResult, false)) {
                    std::cout << "Error while running the function:"
                              << algorithm.second << std::endl;
                    return -1;
                }
                std::cout << algorithm.second << " test passed!" << std::endl;
            }
        }
#endif

#if defined ENABLE_LIBLSB1 || defined ENABLE_LIBLSB2 || defined ENABLE_LIBLSB_C1 || defined ENABLE_LIBLSB_C2
        for (const auto &algorithm : algorithms) {
            for (int i = 0; i < NUM_RUNS; ++i) {
                Matrix tempResult(MATRIX_SIZE);
                tempResult.convertDenseToBlkTridiag(BLOCK_SIZE);
                LSB_Res();
                // Run the algorithm
                algorithm.first(inputMatrix, tempResult, MATRIX_SIZE,
                                BLOCK_SIZE, IS_SYMMETRIC, SAVE_OFF_DIAG);
                LSB_Rec(i);
            }
        }
#else
#endif

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

#if defined ENABLE_LIBLSB1 || defined ENABLE_LIBLSB2 || defined ENABLE_LIBLSB_C1 || defined ENABLE_LIBLSB_C2
    LSB_Finalize();
#endif

    MPI_Finalize();
    return 0;
}
