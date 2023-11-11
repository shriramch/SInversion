#include <iostream>
#include <assert.h>
#include <mpi.h>
#include <cblas.h>
#include <lapacke.h>

using namespace std;

/* Symmetric matrix representation
   n: size of the matrix
   blockSize: size of the block
   mat: matrix in 1D array; 
   mdiag: main diagonal blocks in 1D array; 
   updiag: upper diagonal blocks in 1D array; 
   lodiag: lower diagonal blocks in 1D array
*/
class Matrix {
    int matrixSize, blockSize; 
    float *mat; // indexing: [row][column] = [i][j] = i*blockSize + j
    bool allclose(const float* a, const float* b, std::size_t size, double rtol = 1e-3, double atol = 1e-5, bool isPrint=true);
    
public:
    float *mdiag, *updiag, *lodiag; // indexing: [b][i][j] = b*blockSize*blockSize + i*blockSize + j
    Matrix(int);
    Matrix(int, float *);
    // Copy constructor
    Matrix(const Matrix& other);
    // Copy assignment operator
    Matrix& operator=(const Matrix& other);
    void copyMatrixData(const Matrix& other);
    float* getMat();
    bool compareDiagonals(const Matrix& other, bool);
    void convertDenseToBlkTridiag(const int);
    void convertBlkTridiagToDense();
    void printM();
    void printB();
    void getBlockSizeAndMatrixSize(int &, int &);
    void mmmBLAS(int , float *, float *, float *);
    void invBLAS(int n, const float *A, float *result); // correct
    void mmAdd(int n, float *A, float *B, float *result);
    void mmSub(int n, float *A, float *B, float *result);
    void transposeBLAS(int n, float *A, float *result);
    ~Matrix();
};

void rgf2sided(Matrix &A, Matrix &G, bool sym_mat = false, bool save_off_diag = true
               );
void rgf2sided_upperprocess(Matrix &A, Matrix &G, int nblocks_2, bool sym_mat = false,
                            bool save_off_diag = true);
void rgf2sided_lowerprocess(Matrix &A, Matrix &G, int nblocks_2, bool sym_mat = false,
                            bool save_off_diag = true);
