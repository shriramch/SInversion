#include "rgf2.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>
Matrix generateRandomMat(int matrixSize, bool isSymmetric = false, int seed = 0);
Matrix generateBandedDiagonalMatrix(int matrixSize, int matriceBandwidth, bool isSymmetric=false, int seed=0);
Matrix generateFixedMatrixOfSize8();
void transformToSymmetric(Matrix& A);
void makeMatrixBanded(int matrixSize, int matriceBandwidth, float* matrix);