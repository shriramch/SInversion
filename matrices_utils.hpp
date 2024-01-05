#pragma once
#include <assert.h>
#include <cblas.h>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <lapacke.h>

/**
 *  Symmetric matrix representation
 *  n: size of the matrix
 *  blockSize: size of the block
 *  mat: matrix in 1D array;
 *  mdiag: main diagonal blocks in 1D array;
 *  updiag: upper diagonal blocks in 1D array;
 *  lodiag: lower diagonal blocks in 1D array
 */

class Matrix {
    int matrixSize, blockSize;
    float *mat;  // keep the mat field for future use. not used in this project anymore
    bool allclose(const float *a, const float *b, std::size_t size,
                  double rtol = 1e-3, double atol = 1e-5, bool isPrint = true);

public:
    float *mdiag, *updiag, *lodiag;
    Matrix(int);
    Matrix(int, float *);
    Matrix(const Matrix &other);
    Matrix &operator=(const Matrix &other);
    void copyMatrixData(const Matrix &other);
    bool compareDiagonals(const Matrix &other, bool);
    void convertDenseToBlkTridiag(const int);
    void printM();
    void printB();
    void getBlockSizeAndMatrixSize(int &, int &);
    void mmmBLAS(int, float *, float *, float *);
    void invBLAS(int n, const float *A, float *result); // correct
    void mmAdd(int n, float *A, float *B, float *result);
    void mmSub(int n, float *A, float *B, float *result);
    void matScale(int n, float *A, int k, float *result);
    void transposeBLAS(int n, float *A, float *result);
    ~Matrix();
};

Matrix generateRandomMat(int matrixSize, bool isSymmetric = false,
                         int seed = 0);
Matrix generateBandedDiagonalMatrix(int matrixSize, int matriceBandwidth,
                                    bool isSymmetric = false, int seed = 0);
