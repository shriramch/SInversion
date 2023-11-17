#include "rgf1.hpp"

Matrix::Matrix(int N) {
    n = N;
    mat = new float[N * N];
    for (int i = 0; i < N * N; ++i) {
        mat[i] = 0;
    }
}

Matrix::Matrix(int N, float *newMat) {
    n = N;
    mat = new float[N * N];
    for (int i = 0; i < N * N; ++i) {
        mat[i] = newMat[i];
    }
}

// Matrix::Matrix(float *m, float *up, float *lo) {
//     mdiag = m;
//     updiag = up;
//     lodiag = lo;
// }

void Matrix::DensetoB3D(int blockSize) {
    B = blockSize;
    assert(n % blockSize == 0);
    int numBlocks = n / blockSize;
    mdiag = new float[numBlocks * B * B];
    updiag = new float[(numBlocks - 1) * B * B];
    lodiag = new float[(numBlocks - 1) * B * B];
    for (int b = 0; b < numBlocks; ++b) {
        for (int i = 0; i < blockSize; ++i) {
            for (int j = 0; j < blockSize; ++j) {
                mdiag[(b * blockSize * blockSize) + (i * blockSize) + j] =
                    mat[(b * blockSize + i) * n + (b * blockSize + j)];
            }
        }
    }

    for (int b = 0; b < numBlocks - 1; ++b) {
        for (int i = 0; i < blockSize; ++i) {
            for (int j = 0; j < blockSize; ++j) {
                updiag[b * blockSize * blockSize + i * blockSize + j] =
                    mat[(b * blockSize + i) * n + ((b + 1) * blockSize + j)];
                lodiag[b * blockSize * blockSize + i * blockSize + j] =
                    mat[((b + 1) * blockSize + i) * n + (b * blockSize + j)];
            }
        }
    }
}

void Matrix::B3DtoDense() {
    int numBlocks = n / B;
    for (int b = 0; b < numBlocks; ++b) {
        for (int i = 0; i < B; ++i) {
            for (int j = 0; j < B; ++j) {
                // bth, b * B + i, b * B + j
                mat[(b * B + i) * n + (b * B + j)] =
                    mdiag[b * B * B + i * B + j];
            }
        }
    }
    for (int b = 0; b < numBlocks - 1; ++b) {
        for (int i = 0; i < B; ++i) {
            for (int j = 0; j < B; ++j) {
                // bth, b * B + i, b * B + j
                mat[(b * B + i) * n + (b * B + j + B)] =
                    updiag[b * B * B + i * B + j];
                mat[(b * B + i + B) * n + (b * B + j)] =
                    lodiag[b * B * B + i * B + j];
            }
        }
    }
}

void Matrix:: printM() {

    std::cout << std::fixed << std::setprecision(8);

    cout << "Matrix: " << endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout <<setw(8)<< mat[i * n + j] << " ";
        }
        cout << endl;
    }
}

void Matrix::printB() {

    std::cout << std::fixed << std::setprecision(8);
    
    cout << "Main diagonal: " << endl;
    for (int i = 0; i < n / B; ++i) {
        for (int j = 0; j < B; ++j) {
            for (int k = 0; k < B; ++k) {
                cout << mdiag[i * B * B + j * B + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << "Upper diagonal: " << endl;
    for (int i = 0; i < n / B - 1; ++i) {
        for (int j = 0; j < B; ++j) {
            for (int k = 0; k < B; ++k) {
                cout << updiag[i * B * B + j * B + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << "Lower diagonal: " << endl;
    for (int i = 0; i < n / B - 1; ++i) {
        for (int j = 0; j < B; ++j) {
            for (int k = 0; k < B; ++k) {
                cout << lodiag[i * B * B + j * B + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

void print(float *A, int N) {
    cout << "Matrix: " << endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << A[i * N + j] << " ";
        }
        cout << endl;
    }
}

void Matrix::get(int &t, int &u) { t = B, u = n; }

void rgf(Matrix &A, Matrix &G, bool sym_mat = false,
         bool save_off_diag = true) {

    int blocksize, matrix_dim;

    A.get(blocksize, matrix_dim);

    mat_INV(blocksize, A.mdiag, G.mdiag);


    int num_blocks = matrix_dim / blocksize;

    for (int i = 1; i < num_blocks; ++i) {

        float *AAi = new float[blocksize * blocksize], *AGi = new float[blocksize * blocksize];
        
        MMM_BLAS(blocksize, &(A.lodiag[(i - 1)*blocksize*blocksize]), &(G.mdiag[(i - 1)*blocksize*blocksize]), AGi);
        MMM_BLAS(blocksize, AGi, &(A.updiag[(i - 1)*blocksize*blocksize]), AAi);
        mat_SUB(blocksize, &(A.mdiag[i*blocksize*blocksize]), AAi, AGi);
        mat_INV(blocksize, AGi, &(G.mdiag[i*blocksize*blocksize]));

        free(AAi), free(AGi);
    }


    for (int i = num_blocks - 2; i >= 0; --i) {

        float *Glf = new float[blocksize * blocksize], *Glf1 = new float[blocksize * blocksize];
        
        MMM_BLAS(blocksize, &(G.mdiag[(i + 1)*blocksize*blocksize]), &(A.lodiag[i*blocksize*blocksize]), Glf1);
        MMM_BLAS(blocksize, Glf1, &(G.mdiag[i*blocksize*blocksize]), Glf);


        if (save_off_diag) {

            matK(blocksize, Glf, -1, &(G.lodiag[i*blocksize*blocksize]));
            
            if (sym_mat) {
                
                matT(blocksize, &(G.lodiag[i*blocksize*blocksize]), &(G.updiag[i*blocksize*blocksize]));
            } 
            else {

                float *Guf = new float[blocksize * blocksize], *Guf1 = new float[blocksize * blocksize];
               
                MMM_BLAS(blocksize, &(A.updiag[i*blocksize*blocksize]), &(G.mdiag[(i + 1)*blocksize*blocksize]), Guf1);
                MMM_BLAS(blocksize, &(G.mdiag[i*blocksize*blocksize]), Guf1, Guf);
                matK(blocksize, Guf, -1, &(G.updiag[i*blocksize*blocksize]));

                free(Guf), free(Guf1);
            }
        }

        MMM_BLAS(blocksize, &(A.updiag[i*blocksize*blocksize]), Glf, Glf1);
        MMM_BLAS(blocksize, &(G.mdiag[i*blocksize*blocksize]), Glf1, Glf);
        mat_ADD(blocksize, &(G.mdiag[i*blocksize*blocksize]), Glf, &(G.mdiag[i*blocksize*blocksize]));

        free(Glf), free(Glf1);
    }
}

// int main() {
//     float *t = new float[16];
//     for (int i = 0; i < 4; ++i) {
//         for (int j = max(i - 1, 0); j < min(i + 2, 4); ++j) {
//             t[i*4 + j] = 5;
//         }
//     }
//     Matrix m(4, t), f(4);
//     m.printM();
//     m.DensetoB3D(2);
//     f.DensetoB3D(2);
//     m.printB();
//     rgf(m, f);
//     // f.printB();
//     f.B3DtoDense();
//     f.printM();
//     return 0;
// }

void MMM_BLAS(int n, float *A, float *B, float *result) {
    const CBLAS_ORDER order = CblasRowMajor;
    const CBLAS_TRANSPOSE transA = CblasNoTrans;
    const CBLAS_TRANSPOSE transB = CblasNoTrans;
    const float alpha = 1.0;
    const float beta = 0.0;
    float *res_temp = (float *)malloc(n * n * sizeof(float));

    cblas_sgemm(order, transA, transB, n, n, n, alpha, A, n, B, n, beta,
                res_temp, n);
    memcpy(result, res_temp, n * n * sizeof(float));
    free(res_temp);
}

void MMM_noob(int n, float *A, float *B, float *result) {

    float *res_temp = (float *)malloc(n * n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                res_temp[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
    memcpy(result, res_temp, n * n * sizeof(float));
    free(res_temp);
}

void mat_INV(int n, const float *A, float *result) {

    int *ipiv = (int *)malloc(n * sizeof(int));
    memcpy(result, A, n * n * sizeof(float));

    LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, result, n, ipiv);

    LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, result, n, ipiv);

    free(ipiv);
}

void mat_SUB(int n, float *A, float *B, float *result) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i * n + j] = A[i * n + j] - B[i * n + j];
        }
    }
}

void mat_ADD(int n, float *A, float *B, float *result) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i * n + j] = A[i * n + j] + B[i * n + j];
        }
    }
}

void matK(int n, float *A, int k, float *result) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i * n + j] = A[i * n + j] * k;
        }
    }
}

void matT(int n, float *A, float *result) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i * n + j] = A[j * n + i];
        }
    }
}