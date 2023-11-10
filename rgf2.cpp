#include "rgf2.hpp"

Matrix::~Matrix() {
    delete[] mat;
    delete[] mdiag;
    delete[] updiag;
    delete[] lodiag;
}

/* Zero-initialize matrix  */
Matrix::Matrix(int N) {
    matrixSize = N;
    mat = new float[N * N]();
    blockSize = 1;
    mdiag = NULL;
    updiag = NULL;
    lodiag = NULL;
}

/* initlize matrix with size N and values newMat */
Matrix::Matrix(int N, float *newMat) {
    matrixSize = N;
    mat = new float[N * N];
    memcpy(mat, newMat, N * N * sizeof(float));
    blockSize = 1;
    mdiag = NULL;
    updiag = NULL;
    lodiag = NULL;
}

// convert back to show the result 
void convertBlkTridiagToDense() {
    // TODO
}

/* Generate 3 representations */
void Matrix::convertDenseToBlkTridiag(const int blockSize) {
    this->blockSize = blockSize;
    assert(matrixSize % blockSize == 0); // matrixSize must be divisible by blockSize
    int nblocks = matrixSize / blockSize;
    mdiag = new float[nblocks * blockSize * blockSize];
    updiag = new float[(nblocks - 1) * blockSize * blockSize];
    lodiag = new float[(nblocks - 1) * blockSize * blockSize];
    for (int b = 0; b < nblocks; ++b) {
        for (int i = 0; i < blockSize; ++i) {
            for (int j = 0; j < blockSize; ++j) {
                mdiag[b * blockSize * blockSize + i * blockSize + j] =
                    mat[(b * blockSize + i) * matrixSize + (b * blockSize + j)];
            }
        }
    }

    for (int b = 0; b < nblocks - 1; ++b) {
        for (int i = 0; i < blockSize; ++i) {
            for (int j = 0; j < blockSize; ++j) {
                updiag[b * blockSize * blockSize + i * blockSize + j] =
                    mat[(b * blockSize + i) * matrixSize + ((b + 1) * blockSize + j)];
                lodiag[b * blockSize * blockSize + i * blockSize + j] =
                    mat[((b + 1) * blockSize + i) * matrixSize + (b * blockSize + j)];
            }
        }
    }
}

void Matrix::printM() {
    cout << "Matrix: " << endl;
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            cout << mat[i * matrixSize + j] << " ";
        }
        cout << endl;
    }
}

void Matrix::printB() {
    cout << "Main diagonal: " << endl;
    for (int i = 0; i < matrixSize / blockSize; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            for (int k = 0; k < blockSize; ++k) {
                cout << mdiag[i * blockSize * blockSize + j * blockSize + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << "Upper diagonal: " << endl;
    for (int i = 0; i < matrixSize / blockSize - 1; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            for (int k = 0; k < blockSize; ++k) {
                cout << updiag[i * blockSize * blockSize + j * blockSize + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << "Lower diagonal: " << endl;
    for (int i = 0; i < matrixSize / blockSize - 1; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            for (int k = 0; k < blockSize; ++k) {
                cout << lodiag[i * blockSize * blockSize + j * blockSize + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

void Matrix::getBlockSizeAndMatrixSize(int &outBlockSize, int &outMatrixSize) {
    outBlockSize = blockSize;
    outMatrixSize = matrixSize;
}

float* Matrix::getMat() {
    return mat;
}

void rgf2sided(Matrix &A, Matrix &G, bool sym_mat , bool save_off_diag 
               ) {
    int processRank, blockSize, matrixSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    int nblocks = matrixSize / blockSize;
    int nblocks_2 = nblocks / 2;

    if (processRank == 0) {
        rgf2sided_upperprocess(A, G, nblocks_2, sym_mat, save_off_diag);
        MPI_Recv((void *)(G.mdiag + nblocks_2 * matrixSize * matrixSize), 
                (nblocks_2) * matrixSize * matrixSize, MPI_FLOAT, 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv((void *)(G.updiag + nblocks_2 * matrixSize * matrixSize), 
                (nblocks_2) * matrixSize * matrixSize, MPI_FLOAT, 1, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv((void *)(G.lodiag + nblocks_2 * matrixSize * matrixSize), 
                (nblocks_2 ) * matrixSize * matrixSize, MPI_FLOAT, 1, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
    } else if (processRank == 1) {
        rgf2sided_lowerprocess(A, G, nblocks - nblocks_2, sym_mat, save_off_diag);
        MPI_Send((const void *)(G.mdiag + nblocks_2 * matrixSize * matrixSize),
                 (nblocks - nblocks_2) * matrixSize * matrixSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        MPI_Send((const void *)(G.updiag + nblocks_2 * matrixSize * matrixSize),
                 (nblocks - nblocks_2 - 1) * matrixSize * matrixSize, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        MPI_Send((const void *)(G.lodiag + nblocks_2 * matrixSize * matrixSize),
                 (nblocks - nblocks_2 - 1) * matrixSize * matrixSize, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
    }
}

// References based implementation
void rgf2sided_upperprocess(Matrix &A, Matrix &G, int nblocks_2, bool sym_mat,
                            bool save_off_diag)
{
    int processRank, blockSize, matrixSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    
    float *A_diagblk_leftprocess = A.mdiag;  // 0 to nbloks_2-1(included) blocks
    float *A_upperblk_leftprocess = A.updiag;
    float *A_lowerblk_leftprocess = A.lodiag;

    // Zero initialization
    float *G_diagblk_leftprocess = new float[(nblocks_2+1) * blockSize * blockSize]();
    float *G_upperblk_leftprocess = new float[nblocks_2 * blockSize * blockSize]();
    float *G_lowerblk_leftprocess = new float[nblocks_2 * blockSize * blockSize]();

    float *temp_result_1 = new float[blockSize * blockSize]();    
    float *temp_result_2 = new float[blockSize * blockSize]();
    float* zeros = new float[blockSize * blockSize](); 

    // Initialisation of g - invert first block
    A.invBLAS(blockSize, A_diagblk_leftprocess, G_diagblk_leftprocess); 
    
    // Forward substitution
    for (int i = 1; i < nblocks_2; ++i) {
        A.mmmBLAS(blockSize, A_lowerblk_leftprocess + (i-1) * blockSize * blockSize, 
                   G_diagblk_leftprocess + (i-1) * blockSize * blockSize, temp_result_1);
        A.mmmBLAS(blockSize, temp_result_1, A_upperblk_leftprocess + (i-1) * blockSize * blockSize, 
                   temp_result_1); // 
        A.mmSub(blockSize, A_diagblk_leftprocess + i * blockSize * blockSize, temp_result_1, 
                 temp_result_1); 
        A.invBLAS(blockSize, temp_result_1, G_diagblk_leftprocess + i * blockSize * blockSize);
    }

    // Communicate the left connected block and receive the right connected block
    MPI_Send((const void *)(G_diagblk_leftprocess + (nblocks_2-1) * blockSize * blockSize) , blockSize * blockSize, 
             MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    MPI_Recv((void *)(G_diagblk_leftprocess + nblocks_2 * blockSize * blockSize), blockSize * blockSize, 
             MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // Connection from both sides of the full G
    A.mmmBLAS(blockSize, A_lowerblk_leftprocess + (nblocks_2-2)*blockSize*blockSize, 
               G_diagblk_leftprocess + (nblocks_2-2)*blockSize*blockSize, temp_result_1);
    A.mmmBLAS(blockSize, temp_result_1, A_upperblk_leftprocess + (nblocks_2-2)*blockSize*blockSize, 
               temp_result_1);
    A.mmmBLAS(blockSize, A_upperblk_leftprocess + (nblocks_2-1)*blockSize*blockSize, 
               G_diagblk_leftprocess + (nblocks_2)*blockSize*blockSize, temp_result_2);
    A.mmmBLAS(blockSize, temp_result_2, A_lowerblk_leftprocess + (nblocks_2-1)*blockSize*blockSize, 
               temp_result_2);
    A.mmSub(blockSize, A_diagblk_leftprocess + (nblocks_2-1)*blockSize*blockSize, temp_result_1, 
             temp_result_1);
    A.mmSub(blockSize, temp_result_1, temp_result_2, temp_result_1);
    A.invBLAS(blockSize, temp_result_1, G_diagblk_leftprocess + (nblocks_2-1)*blockSize*blockSize);

    // Compute the shared off-diagonal upper block
    A.mmmBLAS(blockSize, G_diagblk_leftprocess + nblocks_2 * blockSize * blockSize, 
               A_lowerblk_leftprocess + (nblocks_2-1) * blockSize * blockSize, 
               temp_result_1);
    A.mmmBLAS(blockSize, temp_result_1, G_diagblk_leftprocess + (nblocks_2-1) * blockSize * blockSize, 
               temp_result_1);

    A.mmSub(blockSize, zeros, temp_result_1, G_lowerblk_leftprocess + (nblocks_2-1) * blockSize * blockSize);
    if (sym_mat) {
        // matrix transpose
        cblas_somatcopy(CblasRowMajor, CblasTrans, blockSize, blockSize, 1.0f, G_lowerblk_leftprocess + (nblocks_2-1)*blockSize*blockSize, blockSize, G_upperblk_leftprocess + (nblocks_2-1)*blockSize*blockSize, blockSize); 
    }
    else {
        A.mmmBLAS(blockSize, G_diagblk_leftprocess + (nblocks_2-1) * blockSize * blockSize, 
               A_upperblk_leftprocess + (nblocks_2-1) * blockSize * blockSize, 
               temp_result_1);
        A.mmmBLAS(blockSize, temp_result_1, G_diagblk_leftprocess + (nblocks_2) * blockSize * blockSize, 
               temp_result_1);
        A.mmSub(blockSize, zeros, temp_result_1, G_upperblk_leftprocess + (nblocks_2-1) * blockSize * blockSize);
    }

    // Backward substitution
    for (int i = nblocks_2 - 2; i >= 0; i -= 1) {
        float *g_ii = G_diagblk_leftprocess + i * blockSize * blockSize;
        float *G_lowerfactor = new float[blockSize * blockSize];
        A.mmmBLAS(blockSize, G_diagblk_leftprocess+(i+1)*blockSize*blockSize, A_lowerblk_leftprocess+i*blockSize*blockSize, temp_result_1);
        A.mmmBLAS(blockSize, temp_result_1, g_ii, G_lowerfactor);

        if (save_off_diag) {
            A.mmSub(blockSize, zeros, G_lowerfactor, G_lowerblk_leftprocess+i*blockSize*blockSize);
            if(sym_mat) {
                // matrix transpose
                cblas_somatcopy(CblasRowMajor, CblasTrans, blockSize, blockSize, 1.0f, G_lowerblk_leftprocess + (i)*blockSize*blockSize, blockSize, G_upperblk_leftprocess + (i)*blockSize*blockSize, blockSize);
            }
            else {
                A.mmmBLAS(blockSize, g_ii, A_upperblk_leftprocess+i*blockSize*blockSize, temp_result_1);
                A.mmmBLAS(blockSize, temp_result_1, G_diagblk_leftprocess+(i+1)*blockSize*blockSize, temp_result_1);
                A.mmSub(blockSize, zeros, temp_result_1, G_upperblk_leftprocess+i*blockSize*blockSize);
            }
        }
        A.mmmBLAS(blockSize, g_ii, A_upperblk_leftprocess+i*blockSize*blockSize, temp_result_1);
        A.mmmBLAS(blockSize, temp_result_1, G_lowerfactor, temp_result_1);
        A.mmAdd(blockSize, g_ii, temp_result_1, G_diagblk_leftprocess+i*blockSize*blockSize);
    }

    memcpy(G.mdiag, G_diagblk_leftprocess, nblocks_2 * blockSize * blockSize * sizeof(float));
    memcpy(G.updiag, G_upperblk_leftprocess, nblocks_2 * blockSize * blockSize * sizeof(float));
    memcpy(G.lodiag, G_lowerblk_leftprocess, nblocks_2 * blockSize * blockSize * sizeof(float));
    
    // G.printB();

    delete[] G_diagblk_leftprocess;
    delete[] G_upperblk_leftprocess;
    delete[] G_lowerblk_leftprocess;
    delete[] temp_result_1;
    delete[] temp_result_2;
    delete[] zeros;

    return;
}


void rgf2sided_lowerprocess(Matrix &A, Matrix &G, int nblocks_2, bool sym_mat,
                            bool save_off_diag)
{
    int processRank, blockSize, matrixSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    
    float *A_diagblk_rightprocess = A.mdiag + nblocks_2 * blockSize * blockSize;  // nblocks_2 to nblocks-1(included) blocks
    float *A_upperblk_rightprocess = A.updiag + (nblocks_2-1) * blockSize * blockSize; // nblocks_2 -1 to nblocks - 2; note updiag is one smaller than mdiag; A_diagblk_rightprocess and A_upperblk_rightprocess is of same size
    float *A_lowerbk_rightprocess = A.lodiag + (nblocks_2-1) * blockSize * blockSize; 
    // to check correctness
    float *G_diagblk_rightprocess = new float[(nblocks_2+1) * blockSize * blockSize]();
    float *G_upperblk_rightprocess = new float[nblocks_2 * blockSize * blockSize]();
    float *G_lowerblk_rightprocess = new float[nblocks_2 * blockSize * blockSize]();

    float *temp_result_1 = new float[blockSize * blockSize]();    
    float *temp_result_2 = new float[blockSize * blockSize]();
    float* zeros = new float[blockSize * blockSize](); 

    // Initialisation of g - invert first block
    A.invBLAS(blockSize, A_diagblk_rightprocess + nblocks_2, G_diagblk_rightprocess + nblocks_2 ); 
    
    // Forward substitution
    for (int i = nblocks_2-1; i >= 1; i -= 1) {
        A.mmmBLAS(blockSize, A_upperblk_rightprocess + i * blockSize * blockSize, 
                   G_diagblk_rightprocess + (i+1) * blockSize * blockSize, temp_result_1);
        A.mmmBLAS(blockSize, temp_result_1, A_lowerbk_rightprocess + i * blockSize * blockSize, 
                   temp_result_1);
        A.mmSub(blockSize, A_diagblk_rightprocess + (i-1) * blockSize * blockSize, temp_result_1, 
                 temp_result_1); 
        A.invBLAS(blockSize, temp_result_1, G_diagblk_rightprocess + i * blockSize * blockSize);
    }

    // Communicate the right connected block and receive the right connected block
    MPI_Recv((void *)(G_diagblk_rightprocess), blockSize * blockSize, 
             MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send((const void *)(G_diagblk_rightprocess + 1 * blockSize * blockSize) , blockSize * blockSize, 
             MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
   
    // Connection from both sides of the full G
    A.mmmBLAS(blockSize, A_lowerbk_rightprocess, 
               G_diagblk_rightprocess, temp_result_1);
    A.mmmBLAS(blockSize, temp_result_1, A_upperblk_rightprocess, 
               temp_result_1);
    A.mmmBLAS(blockSize, A_upperblk_rightprocess + (1)*blockSize*blockSize, 
               G_diagblk_rightprocess + (2)*blockSize*blockSize, temp_result_2);
    A.mmmBLAS(blockSize, temp_result_2, A_lowerbk_rightprocess + (1)*blockSize*blockSize, 
               temp_result_2);
    A.mmSub(blockSize, A_diagblk_rightprocess, temp_result_1, 
             temp_result_1);
    A.mmSub(blockSize, temp_result_1, temp_result_2, temp_result_1);
    A.invBLAS(blockSize, temp_result_1, G_diagblk_rightprocess + (1)*blockSize*blockSize);

    // Compute the shared off-diagonal upper block
    A.mmmBLAS(blockSize, G_diagblk_rightprocess + 1 * blockSize * blockSize, 
               A_lowerbk_rightprocess, 
               temp_result_1);
    A.mmmBLAS(blockSize, temp_result_1, G_diagblk_rightprocess, 
               temp_result_1);

    if (sym_mat) {
        // matrix transpose
        cblas_somatcopy(CblasRowMajor, CblasTrans, blockSize, blockSize, 1.0f, G_lowerblk_rightprocess, blockSize, G_upperblk_rightprocess, blockSize); 
    }
    else {
        A.mmmBLAS(blockSize, G_diagblk_rightprocess, 
               A_upperblk_rightprocess, 
               temp_result_1);
        A.mmmBLAS(blockSize, temp_result_1, G_diagblk_rightprocess + 1 * blockSize*blockSize, 
               G_upperblk_rightprocess);
    }

    // Backward substitution
    for (int i = 1; i < nblocks_2; ++i) {
        float *g_ii = G_diagblk_rightprocess + (i+1) * blockSize * blockSize;
        float *G_lowerfactor = new float[blockSize * blockSize];
        A.mmmBLAS(blockSize, g_ii, A_lowerbk_rightprocess + i * blockSize * blockSize, temp_result_1);
        A.mmmBLAS(blockSize, temp_result_1, G_diagblk_rightprocess + i * blockSize * blockSize, G_lowerfactor);
        if (save_off_diag) {
            A.mmSub(blockSize, zeros, G_lowerfactor, G_lowerblk_rightprocess + i * blockSize * blockSize);
            if (sym_mat) {
                // matrix transpose
                cblas_somatcopy(CblasRowMajor, CblasTrans, blockSize, blockSize, 1.0f, G_lowerblk_rightprocess + i * blockSize * blockSize, blockSize, G_upperblk_rightprocess + i * blockSize * blockSize, blockSize); 
            }
            else {
                A.mmmBLAS(blockSize, G_diagblk_rightprocess + i*blockSize*blockSize, 
                    A_upperblk_rightprocess + i*blockSize*blockSize, 
                    temp_result_1);
                A.mmmBLAS(blockSize, temp_result_1, g_ii, 
                    temp_result_1);
                A.mmSub(blockSize, zeros, temp_result_1, G_upperblk_rightprocess + i * blockSize * blockSize);
            }
        }
        A.mmmBLAS(blockSize, G_lowerfactor, 
                    A_upperblk_rightprocess + i*blockSize*blockSize, 
                    temp_result_1);
        A.mmmBLAS(blockSize, temp_result_1, g_ii, 
                    temp_result_1);
        A.mmAdd(blockSize, g_ii, temp_result_1, G_diagblk_rightprocess + (i+1)*blockSize*blockSize);
    }

    memcpy(G.mdiag, G_diagblk_rightprocess+1, nblocks_2 * blockSize * blockSize * sizeof(float));
    memcpy(G.updiag, G_upperblk_rightprocess, nblocks_2 * blockSize * blockSize * sizeof(float));
    memcpy(G.lodiag, G_lowerblk_rightprocess, nblocks_2 * blockSize * blockSize * sizeof(float));
    
    delete[] G_diagblk_rightprocess;
    delete[] G_upperblk_rightprocess;
    delete[] G_lowerblk_rightprocess;
    delete[] temp_result_1;
    delete[] temp_result_2;
    delete[] zeros;

    return;
}


void Matrix::mmmBLAS(int n, float *A, float *B, float *result) {
    const CBLAS_ORDER order = CblasRowMajor;
    const CBLAS_TRANSPOSE transA = CblasNoTrans;
    const CBLAS_TRANSPOSE transB = CblasNoTrans;
    const float alpha = 1.0;
    const float beta = 0.0;

    cblas_sgemm(order, transA, transB, n, n, n, alpha, A, n, B, n, beta, result, n);
}

void Matrix::mmSub(int n, float *A, float *B, float *result){
    for (int i = 0; i < n * n; ++i) {
        result[i] = A[i] - B[i];
    }
}

void Matrix::mmAdd(int n, float *A, float *B, float *result) {
    for (int i = 0; i < n * n; ++i) {
        result[i] = A[i] + B[i];
    }
}

void Matrix::invBLAS(int n, const float *A, float *result) {

    int *ipiv = (int *)malloc(n * sizeof(int));
    memcpy(result, A, n * n * sizeof(float));

    LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, result, n, ipiv);

    LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, result, n, ipiv);

    free(ipiv);
}


int main(int argc, char **argv) {
    int processRank;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    float *t = new float[16*16]();
    for (int i = 0; i < 16 ; ++i) {
        t[i * 16 + i] = i + 1;
    }
    Matrix A(16, t);
    A.convertDenseToBlkTridiag(2);

    Matrix G(16); // zero initialization, same shape as A
    G.convertDenseToBlkTridiag(2); // G has same blockSize as in A
    rgf2sided(A, G,false, true);
    if(processRank == 0){
        // A.invBLAS(16, A.getMat(), G.getMat());
        // A.convertDenseToBlkTridiag(2);
        // A.printB();
        G.printB();
        // A.printM();
        // G.printM();
    }
    MPI_Finalize();
    /* from the above test, I use the diagnoal matrix, because it is easy to
    invert and check the correctness. This means you don't need to invert A 
    using other method.
    The result shows that the upper part is ok
    some error in lower part
    in order to use G.printM()
    you need to convert it back (which is not implemented yet)
    so use G.printB() to check the result.
    */
}