#include "matrices_utils.hpp"

#include <iostream>
#include <cmath>
#include <functional>
#include <vector>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
int ARGC;
char **ARGV;

// Function pointer type for the algorithm (so is easier to test them, they would all expect a Matrix and return another one)
typedef std::function<void(Matrix&, Matrix&, int, int)> AlgorithmFunction; // input as originalMatrix, resultMatrix, MATRIX_SIZE, BLOCK_SIZE

// Base RGF 2-Sided algorithm
void rgf2sidedAlgorithm(Matrix& input, Matrix& result, int matrixSize, int blockSize) {
    rgf2sided(input, result, false, true);
}

// Another algorithm to test
// void anotherAlgorithmconst Matrix& input, Matrix& result, int matrixSize, int blockSize) {
//     // Implement your other algorithm here
//     // ...
// }

int main(int argc, char **argv) {
    // Note, skipping the whole MPI thing here as it will be done internally in each function if needed
    ARGC = argc;
    ARGV = argv;
    int processRank;
    MPI_Init(&ARGC, &ARGV);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    
    // Init CONST
    int MATRIX_SIZE = 32;
    int BLOCK_SIZE = 4; // MATRIX_SIZE should be divisible by this
    int NUM_RUNS = 10;

    // Vector of algorithm functions and their names
    std::vector<std::pair<AlgorithmFunction, std::string>> algorithms = {
        {rgf2sidedAlgorithm, "rgf2sidedAlgorithm"}
        // {anotherAlgorithm, "Another Algorithm"}
        // Add more algorithms as needed
    };

    // Generate random input matrix
    // Case Simple Diagonal matrix works
    // float *t = new float[MATRIX_SIZE*MATRIX_SIZE]();
    // for (int i = 0; i < MATRIX_SIZE ; ++i) {
    //     t[i * MATRIX_SIZE + i] = i + 1;
    // }
    // Matrix inputMatrix(MATRIX_SIZE, t);
    // The randomly generated one does not work
    Matrix inputMatrix = generateRandomMat(MATRIX_SIZE, true, 42);
    inputMatrix.convertDenseToBlkTridiag(BLOCK_SIZE);
    
    // Calculate base result using base algorithm
    Matrix baseResultMatrix(MATRIX_SIZE); // zero initialization, same shape as inputMatrix
    baseResultMatrix.convertDenseToBlkTridiag(BLOCK_SIZE); // G has same blockSize as inputMatrix
    rgf2sidedAlgorithm(inputMatrix, baseResultMatrix, MATRIX_SIZE, BLOCK_SIZE);// TODO, modify with 1 sided
    // Check correctness for each algorithm
    for (const auto& algorithm : algorithms) {
        Matrix tempResult(MATRIX_SIZE); // zero initialization, same shape as inputMatrix
        tempResult.convertDenseToBlkTridiag(4); // G has same blockSize as in inputMatrix
    
        algorithm.first(inputMatrix, tempResult, MATRIX_SIZE, BLOCK_SIZE);
        if(processRank == 0){
            if ( !baseResultMatrix.compareDiagonals(tempResult)){
                std::cout << "Error while running the function:" << algorithm.second << std::endl;
                return -1;
            }
        }
    }
    
    // Run all the functions for x times and measure the time
    // create just a single output, as i have already checked correctness, now just need the timing of multiple runs
    Matrix tempResult(MATRIX_SIZE); // zero initialization, same shape as inputMatrix
    tempResult.convertDenseToBlkTridiag(4); // G has same blockSize as in inputMatrix
    for (const auto& algorithm : algorithms) {
        for (int i = 0; i < NUM_RUNS; ++i) {
            
            auto start = clock(); //decide whether to use this or just clock()

            // Run the algorithm
            algorithm.first(inputMatrix, tempResult, MATRIX_SIZE, BLOCK_SIZE);
            
            auto end = clock();
            auto duration = MAX(1, (end - start));
            // Output the time taken for each function
            if(processRank == 0){
                std::cout << algorithm.second << " Time: " << duration << std::endl;
            }
        }
    }

    MPI_Finalize();
    return 0;
}