#include "rgf2.hpp"
#include <fstream>

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
void Matrix::convertBlkTridiagToDense() {
    assert(matrixSize % blockSize == 0); // matrixSize must be divisible by blockSize
    int nblocks = matrixSize / blockSize;
    // Assume it is called after initialization of mdiag, updiag, lodiag
    for (int b = 0; b < nblocks; ++b) {
        for (int i = 0; i < blockSize; ++i) {
            for (int j = 0; j < blockSize; ++j) {
                mat[(b * blockSize + i) * matrixSize + (b * blockSize + j)]
                = mdiag[b * blockSize * blockSize + i * blockSize + j];
            }
        }
    }

    for (int b = 0; b < nblocks - 1; ++b) {
        for (int i = 0; i < blockSize; ++i) {
            for (int j = 0; j < blockSize; ++j) {
                mat[(b * blockSize + i) * matrixSize + ((b + 1) * blockSize + j)] = 
                updiag[b * blockSize * blockSize + i * blockSize + j];
                mat[((b + 1) * blockSize + i) * matrixSize + (b * blockSize + j)] = 
                lodiag[b * blockSize * blockSize + i * blockSize + j];
            }
        }
    }
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

// Note, we expect the same size for both the lists
bool Matrix::allclose(const float* a, const float* b, std::size_t size, double rtol, double atol) {
    for (std::size_t i = 0; i < size; ++i) {
        std::cout << a[i] << " vs " << b[i] << std::endl;
        if (std::abs(a[i] - b[i]) > (atol + rtol * std::abs(b[i]))) {
            return false;  // Elements are not almost equal
        }
    }

    return true;  // All elements are almost equal
}

bool Matrix::compareDiagonals(const Matrix& other) {
    // Compare main diagonal (mdiag)
    if (!allclose(mdiag, other.mdiag, matrixSize * blockSize, 1e-5, 1e-8)) {
        std::cout << "Main diagonal not equal." << std::endl;
        return false;
    }

    // Compare upper diagonal (updiag)
    if (!allclose(updiag, other.updiag, (matrixSize - blockSize) * blockSize, 1e-5, 1e-8)) {
        std::cout << "Upper diagonal not equal." << std::endl;
        return false;
    }

    // Compare lower diagonal (lodiag)
    if (!allclose(lodiag, other.lodiag, (matrixSize - blockSize) * blockSize, 1e-5, 1e-8)) {
        std::cout << "Lower diagonal not equal." << std::endl;
        return false;
    }
    return true;
}
void write_array(std::string filepath, float* array, int size) {
    std::ofstream outputFile;
    outputFile.open(filepath, std::ios::app); // Open the file in append mode
    if (outputFile.is_open()) {
        for(int i = 0; i < size; ++i) {
            for(int j = 0; j < size; ++j) {
                outputFile << array[i * size + j] << " ";
            }
            outputFile << std::endl;
        }
        outputFile << std::endl;
        outputFile.close();
        // std::cout << "Matrix appended to file." << std::endl;
    } else {
        std::cout << "Unable to open the file." << std::endl;
    }
}
void rgf2sided(Matrix &A, Matrix &G, bool sym_mat , bool save_off_diag 
               ) {
    int processRank, blockSize, matrixSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    int nblocks = matrixSize / blockSize; // assume divisible
    int nblocks_2 = nblocks / 2; // assume divisible

    if (processRank == 0) {
        rgf2sided_upperprocess(A, G, nblocks_2, sym_mat, save_off_diag);
        
        // float buffer = 0.0;
        // MPI_Recv((void *)&buffer, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // printf("Received %f\n", buffer);
        MPI_Recv((void *)(G.mdiag + nblocks_2 * blockSize * blockSize), 
                (nblocks_2) * blockSize * blockSize, MPI_FLOAT, 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv((void *)(G.updiag + nblocks_2 * blockSize * blockSize), 
                (nblocks_2-1) * blockSize * blockSize, MPI_FLOAT, 1, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv((void *)(G.lodiag + nblocks_2 * blockSize * blockSize), 
                (nblocks_2-1) * blockSize * blockSize, MPI_FLOAT, 1, 2,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // for(int i = 0; i < nblocks_2-1; ++i) {
        //     write_array("/Users/evan/SInversion/output0.txt", G.lodiag + (nblocks_2+i)*blockSize*blockSize, blockSize);
        // }
        
    } else if (processRank == 1) {
        Matrix G_dup(matrixSize, G.getMat());
        G_dup.convertDenseToBlkTridiag(blockSize);
        rgf2sided_lowerprocess(A, G_dup, nblocks - nblocks_2, sym_mat, save_off_diag);
        // float send_buffer = 100.1;
        // MPI_Send((const void *)&send_buffer, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        // G.printB();
        // should I replicate G?
        // for(int i = 0; i < nblocks_2-1; ++i) {
        //     write_array("/Users/evan/SInversion/output1.txt", G_dup.lodiag + (nblocks_2+i)*blockSize*blockSize, blockSize);
        // }
        MPI_Send((const void *)(G_dup.mdiag + nblocks_2 * blockSize * blockSize),
                 (nblocks - nblocks_2) * blockSize * blockSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        MPI_Send((const void *)(G_dup.updiag + nblocks_2 * blockSize * blockSize),
                 (nblocks - nblocks_2 - 1) * blockSize * blockSize, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        MPI_Send((const void *)(G_dup.lodiag + nblocks_2 * blockSize * blockSize),
                 (nblocks - nblocks_2 - 1) * blockSize * blockSize, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
    }
}


// References based implementation
void rgf2sided_upperprocess(Matrix &A, Matrix &G, int nblocks_2, bool sym_mat,
                            bool save_off_diag)
{
    int blockSize, matrixSize;
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
    float *temp_result_3 = new float[blockSize * blockSize]();
    float *temp_result_4 = new float[blockSize * blockSize]();

    float* zeros = new float[blockSize * blockSize](); 

    // Initialisation of g - invert first block
    A.invBLAS(blockSize, A_diagblk_leftprocess, G_diagblk_leftprocess); 
    std::string filepath("/Users/evan/SInversion/output0.txt");
    // write_array(filepath, G_diagblk_leftprocess, blockSize);

    // Forward substitution
    for (int i = 1; i < nblocks_2; ++i) {
        A.mmmBLAS(blockSize, A_lowerblk_leftprocess + (i-1) * blockSize * blockSize, 
                   G_diagblk_leftprocess + (i-1) * blockSize * blockSize, temp_result_1);
        // write_array(filepath, temp_result_1, blockSize);
        A.mmmBLAS(blockSize, temp_result_1, A_upperblk_leftprocess + (i-1) * blockSize * blockSize, 
                   temp_result_2); // 
        // write_array(filepath, temp_result_2, blockSize);
        A.mmSub(blockSize, A_diagblk_leftprocess + i * blockSize * blockSize, temp_result_2, 
                 temp_result_2); 
        // write_array(filepath, temp_result_2, blockSize);
        A.invBLAS(blockSize, temp_result_2, G_diagblk_leftprocess + i * blockSize * blockSize);
        // write_array(filepath, G_diagblk_leftprocess + i * blockSize * blockSize, blockSize);
    }

    // Communicate the left connected block and receive the right connected block
    MPI_Send((const void *)(G_diagblk_leftprocess + (nblocks_2-1) * blockSize * blockSize) , blockSize * blockSize, 
             MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    MPI_Recv((void *)(G_diagblk_leftprocess + nblocks_2 * blockSize * blockSize), blockSize * blockSize, 
             MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // write_array(filepath, G_diagblk_leftprocess + nblocks_2 * blockSize * blockSize, blockSize);

    // Connection from both sides of the full G
    A.mmmBLAS(blockSize, A_lowerblk_leftprocess + (nblocks_2-2)*blockSize*blockSize, 
               G_diagblk_leftprocess + (nblocks_2-2)*blockSize*blockSize, temp_result_1);
    A.mmmBLAS(blockSize, temp_result_1, A_upperblk_leftprocess + (nblocks_2-2)*blockSize*blockSize, 
               temp_result_2);
    
    A.mmmBLAS(blockSize, A_upperblk_leftprocess + (nblocks_2-1)*blockSize*blockSize, 
               G_diagblk_leftprocess + (nblocks_2)*blockSize*blockSize, temp_result_3);
    A.mmmBLAS(blockSize, temp_result_3, A_lowerblk_leftprocess + (nblocks_2-1)*blockSize*blockSize, 
               temp_result_4);
    A.mmSub(blockSize, A_diagblk_leftprocess + (nblocks_2-1)*blockSize*blockSize, temp_result_2, 
             temp_result_2);
    A.mmSub(blockSize, temp_result_2, temp_result_4, temp_result_2);
    A.invBLAS(blockSize, temp_result_2, G_diagblk_leftprocess + (nblocks_2-1)*blockSize*blockSize);
    // write_array(filepath, G_diagblk_leftprocess + (nblocks_2-1)*blockSize*blockSize, blockSize);

    // Compute the shared off-diagonal upper block
    write_array(filepath, G_diagblk_leftprocess + nblocks_2*blockSize*blockSize, blockSize);
    write_array(filepath, A_lowerblk_leftprocess + (nblocks_2-1)*blockSize*blockSize, blockSize);
    A.mmmBLAS(blockSize, G_diagblk_leftprocess + nblocks_2 * blockSize * blockSize, 
               A_lowerblk_leftprocess + (nblocks_2-1) * blockSize * blockSize, 
               temp_result_1);
    write_array(filepath, temp_result_1, blockSize);
    A.mmmBLAS(blockSize, temp_result_1, G_diagblk_leftprocess + (nblocks_2-1) * blockSize * blockSize, 
               temp_result_2);
    write_array(filepath, temp_result_2, blockSize);
    A.mmSub(blockSize, zeros, temp_result_2, G_lowerblk_leftprocess + (nblocks_2-1) * blockSize * blockSize);
    write_array(filepath, G_lowerblk_leftprocess + (nblocks_2-1) * blockSize * blockSize, blockSize);
    if (sym_mat) {
        // matrix transpose
        cblas_somatcopy(CblasRowMajor, CblasTrans, blockSize, blockSize, 1.0f, G_lowerblk_leftprocess + (nblocks_2-1)*blockSize*blockSize, blockSize, G_upperblk_leftprocess + (nblocks_2-1)*blockSize*blockSize, blockSize); 
        // write_array(filepath, G_upperblk_leftprocess + (nblocks_2-1)*blockSize*blockSize, blockSize);
    }
    else {
        A.mmmBLAS(blockSize, G_diagblk_leftprocess + (nblocks_2-1) * blockSize * blockSize, 
               A_upperblk_leftprocess + (nblocks_2-1) * blockSize * blockSize, 
               temp_result_1);
        A.mmmBLAS(blockSize, temp_result_1, G_diagblk_leftprocess + (nblocks_2) * blockSize * blockSize, 
               temp_result_2);
        A.mmSub(blockSize, zeros, temp_result_2, G_upperblk_leftprocess + (nblocks_2-1) * blockSize * blockSize);
        // write_array(filepath, G_upperblk_leftprocess + (nblocks_2-1)*blockSize*blockSize, blockSize);
    }

    // Backward substitution
    for (int i = nblocks_2 - 2; i >= 0; i -= 1) {
        float *g_ii = G_diagblk_leftprocess + i * blockSize * blockSize;
        // write_array(filepath, g_ii, blockSize);
        float *G_lowerfactor = new float[blockSize * blockSize];
        A.mmmBLAS(blockSize, G_diagblk_leftprocess+(i+1)*blockSize*blockSize, A_lowerblk_leftprocess+i*blockSize*blockSize, temp_result_1);
        A.mmmBLAS(blockSize, temp_result_1, g_ii, G_lowerfactor);
        // write_array(filepath, G_lowerfactor, blockSize);
        if (save_off_diag) {
            A.mmSub(blockSize, zeros, G_lowerfactor, G_lowerblk_leftprocess + i*blockSize*blockSize);
            write_array(filepath, G_lowerblk_leftprocess + i*blockSize*blockSize, blockSize);
            if(sym_mat) {
                // matrix transpose
                cblas_somatcopy(CblasRowMajor, CblasTrans, blockSize, blockSize, 1.0f, G_lowerblk_leftprocess + (i)*blockSize*blockSize, blockSize, G_upperblk_leftprocess + (i)*blockSize*blockSize, blockSize);
                // write_array(filepath, G_upperblk_leftprocess + (i)*blockSize*blockSize, blockSize);
            }
            else {
                A.mmmBLAS(blockSize, g_ii, A_upperblk_leftprocess+i*blockSize*blockSize, temp_result_1);
                A.mmmBLAS(blockSize, temp_result_1, G_diagblk_leftprocess+(i+1)*blockSize*blockSize, temp_result_2);
                A.mmSub(blockSize, zeros, temp_result_2, G_upperblk_leftprocess + i*blockSize*blockSize);
                // write_array(filepath, G_upperblk_leftprocess+i*blockSize*blockSize, blockSize);
            }
        }
        A.mmmBLAS(blockSize, g_ii, A_upperblk_leftprocess+i*blockSize*blockSize, temp_result_1);
        A.mmmBLAS(blockSize, temp_result_1, G_lowerfactor, temp_result_2);
        A.mmAdd(blockSize, g_ii, temp_result_2, G_diagblk_leftprocess+i*blockSize*blockSize);
        // write_array(filepath, G_diagblk_leftprocess+i*blockSize*blockSize, blockSize);
        delete[] G_lowerfactor;
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
    delete[] temp_result_3;
    delete[] temp_result_4;
    delete[] zeros;

    return;
}


void rgf2sided_lowerprocess(Matrix &A, Matrix &G, int nblocks_2, bool sym_mat,
                            bool save_off_diag)
{
    int blockSize, matrixSize;
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

    float *temp_result_3 = new float[blockSize * blockSize]();
    float *temp_result_4 = new float[blockSize * blockSize]();
    float* zeros = new float[blockSize * blockSize](); 

    A.invBLAS(blockSize, A_diagblk_rightprocess + (nblocks_2-1)*blockSize*blockSize, G_diagblk_rightprocess + nblocks_2*blockSize*blockSize); 
    std::string filepath = "/Users/evan/SInversion/output1.txt";
    // write_array(filepath, G_diagblk_rightprocess + nblocks_2*blockSize*blockSize, blockSize);

    // Forward substitution
    for (int i = nblocks_2-1; i >= 1; i -= 1) {
        A.mmmBLAS(blockSize, A_upperblk_rightprocess + i * blockSize * blockSize, 
                   G_diagblk_rightprocess + (i+1) * blockSize * blockSize, temp_result_1);
        
        A.mmmBLAS(blockSize, temp_result_1, A_lowerbk_rightprocess + i * blockSize * blockSize, 
                   temp_result_2);
        
        A.mmSub(blockSize, A_diagblk_rightprocess + (i-1) * blockSize * blockSize, temp_result_2, 
                 temp_result_2); 

        A.invBLAS(blockSize, temp_result_2, G_diagblk_rightprocess + i * blockSize * blockSize);
        
        // write_array(filepath, G_diagblk_rightprocess + i * blockSize * blockSize, blockSize);
    }

    // Communicate the right connected block and receive the right connected block
    
    MPI_Recv((void *)(G_diagblk_rightprocess), blockSize * blockSize, 
             MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send((const void *)(G_diagblk_rightprocess + 1 * blockSize * blockSize) , blockSize * blockSize, 
             MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    // write_array(filepath, G_diagblk_rightprocess, blockSize);

    // Connection from both sides of the full G
    A.mmmBLAS(blockSize, A_lowerbk_rightprocess, 
               G_diagblk_rightprocess, temp_result_1);
    A.mmmBLAS(blockSize, temp_result_1, A_upperblk_rightprocess, 
               temp_result_2);
    A.mmmBLAS(blockSize, A_upperblk_rightprocess + (1)*blockSize*blockSize, 
               G_diagblk_rightprocess + (2)*blockSize*blockSize, temp_result_3);
    A.mmmBLAS(blockSize, temp_result_3, A_lowerbk_rightprocess + (1)*blockSize*blockSize, 
               temp_result_4);
    A.mmSub(blockSize, A_diagblk_rightprocess, temp_result_2, 
             temp_result_2);
    A.mmSub(blockSize, temp_result_2, temp_result_4, temp_result_2);
    A.invBLAS(blockSize, temp_result_2, G_diagblk_rightprocess + (1)*blockSize*blockSize);
    // write_array(filepath, G_diagblk_rightprocess + (1)*blockSize*blockSize, blockSize);

    // Compute the shared off-diagonal upper block
    A.mmmBLAS(blockSize, G_diagblk_rightprocess + 1 * blockSize * blockSize, 
               A_lowerbk_rightprocess, 
               temp_result_1);
    A.mmmBLAS(blockSize, temp_result_1, G_diagblk_rightprocess, 
               G_lowerblk_rightprocess);
    write_array(filepath, G_lowerblk_rightprocess, blockSize);    
    if (sym_mat) {
        // matrix transpose
        cblas_somatcopy(CblasRowMajor, CblasTrans, blockSize, blockSize, 1.0f, G_lowerblk_rightprocess, blockSize, G_upperblk_rightprocess, blockSize); 
        // write_array(filepath, G_upperblk_rightprocess, blockSize);
    }
    else {
        A.mmmBLAS(blockSize, G_diagblk_rightprocess, 
               A_upperblk_rightprocess, 
               temp_result_1);
        A.mmmBLAS(blockSize, temp_result_1, G_diagblk_rightprocess + 1 * blockSize*blockSize, 
               G_upperblk_rightprocess);
        // write_array(filepath, G_upperblk_rightprocess, blockSize);
    }

    // Backward substitution
    for (int i = 1; i < nblocks_2; ++i) {
        float *g_ii = G_diagblk_rightprocess + (i+1) * blockSize * blockSize;
        float *G_lowerfactor = new float[blockSize * blockSize];
        A.mmmBLAS(blockSize, g_ii, A_lowerbk_rightprocess + i * blockSize * blockSize, temp_result_1);
        A.mmmBLAS(blockSize, temp_result_1, G_diagblk_rightprocess + i * blockSize * blockSize, G_lowerfactor);
        if (save_off_diag) {
            A.mmSub(blockSize, zeros, G_lowerfactor, G_lowerblk_rightprocess + i * blockSize * blockSize);
            write_array(filepath, G_lowerblk_rightprocess + i * blockSize * blockSize, blockSize);
            if (sym_mat) {
                // matrix transpose
                cblas_somatcopy(CblasRowMajor, CblasTrans, blockSize, blockSize, 1.0f, G_lowerblk_rightprocess + i * blockSize * blockSize, blockSize, G_upperblk_rightprocess + i * blockSize * blockSize, blockSize); 
                // write_array(filepath, G_upperblk_rightprocess + i * blockSize * blockSize, blockSize);
            }
            else {
                A.mmmBLAS(blockSize, G_diagblk_rightprocess + i*blockSize*blockSize, 
                    A_upperblk_rightprocess + i*blockSize*blockSize, 
                    temp_result_1);
                A.mmmBLAS(blockSize, temp_result_1, g_ii, 
                    temp_result_2);
                A.mmSub(blockSize, zeros, temp_result_2, G_upperblk_rightprocess + i * blockSize * blockSize);
                // write_array(filepath, G_upperblk_rightprocess + i * blockSize * blockSize, blockSize);
            }
        }
        A.mmmBLAS(blockSize, G_lowerfactor, 
                    A_upperblk_rightprocess + i*blockSize*blockSize, 
                    temp_result_1);
        A.mmmBLAS(blockSize, temp_result_1, g_ii, 
                    temp_result_2);
        A.mmAdd(blockSize, g_ii, temp_result_2, G_diagblk_rightprocess + (i+1)*blockSize*blockSize);
        // write_array(filepath, G_diagblk_rightprocess + (i+1)*blockSize*blockSize, blockSize);
    }
    
    memcpy(G.mdiag + nblocks_2*blockSize*blockSize, G_diagblk_rightprocess+blockSize*blockSize, nblocks_2 * blockSize * blockSize * sizeof(float));
    memcpy(G.updiag + (nblocks_2-1)*blockSize*blockSize, G_upperblk_rightprocess, nblocks_2 * blockSize * blockSize * sizeof(float));
    memcpy(G.lodiag + (nblocks_2-1)*blockSize*blockSize, G_lowerblk_rightprocess, nblocks_2 * blockSize * blockSize * sizeof(float));
    
    // for(int i = 0; i < nblocks_2; ++i) {
    //     write_array(filepath, G.lodiag + (nblocks_2-1+i)*blockSize*blockSize, blockSize);
    // }

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


// int main(int argc, char **argv) {
//     int processRank;
//     int MATRIX_SIZE = 32;
//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
//     float *t = new float[MATRIX_SIZE*MATRIX_SIZE]();
//     for (int i = 0; i < MATRIX_SIZE ; ++i) {
//         t[i * MATRIX_SIZE + i] = i + 1;
//     }
//     Matrix A(MATRIX_SIZE, t);
//     A.convertDenseToBlkTridiag(4);

//     Matrix G(MATRIX_SIZE); // zero initialization, same shape as A
//     G.convertDenseToBlkTridiag(4); // G has same blockSize as in A
//     rgf2sided(A, G,false, true);
//     if(processRank == 0){
//         // A.invBLAS(16, A.getMat(), G.getMat());
//         // A.convertDenseToBlkTridiag(2);
//         // A.printB();
//         G.printB();
//         // A.printM();
//         // G.printM();
//     }
//     MPI_Finalize();
//     /* from the above test, I use the diagnoal matrix, because it is easy to
//     invert and check the correctness. This means you don't need to invert A 
//     using other method.
//     The result shows that the upper part is ok
//     some error in lower part
//     in order to use G.printM()
//     you need to convert it back (which is not implemented yet)
//     so use G.printB() to check the result.
//     */
// }