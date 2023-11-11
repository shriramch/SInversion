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

Matrix::Matrix(float *m, float *up, float *lo) {
    mdiag = m;
    updiag = up;
    lodiag = lo;
}

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
                mdiag[b * blockSize * blockSize + i * blockSize + j] =
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

void Matrix::printM() {
    cout << "Matrix: " << endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << mat[i * n + j] << " ";
        }
        cout << endl;
    }
}

void Matrix::printB() {
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

void Matrix::get(int &t, int &u) { t = B, u = n; }

void rgf(Matrix &A, Matrix &G, bool sym_mat = false,
         bool save_off_diag = true) {
    int B, n;
    A.get(B, n);
    mat_INV(B, A.mdiag, G.mdiag);

    int nB = n / B;
    for (int i = 1; i < nB; ++i) {
        float *AAi = new float[B * B], *AGi = new float[B * B];
        MMM_BLAS(B, &(A.lodiag[i - 1]), &(G.mdiag[i - 1]), AGi);
        MMM_BLAS(B, AGi, &(A.updiag[i - 1]), AAi);
        mat_SUB(B, &(A.mdiag[i]), AAi, AGi);
        mat_INV(B, AGi, &(G.mdiag[i]));

        free(AAi), free(AGi);
    }

    for (int i = nB - 2; i >= 0; --i) {
        float *Glf = new float[B * B], *Glf1 = new float[B * B];
        MMM_BLAS(B, &(G.mdiag[i + 1]), &(A.lodiag[i]), Glf1);
        MMM_BLAS(B, Glf1, &(G.mdiag[i]), Glf);

        if (save_off_diag) {
            matK(B, Glf, -1, &(G.lodiag[i]));
            if (sym_mat) {
                matT(B, &(G.lodiag[i]), &(G.updiag[i]));
            } else {
                float *Guf = new float[B * B], *Guf1 = new float[B * B];
                MMM_BLAS(B, &(A.updiag[i]), &(G.mdiag[i + 1]), Guf1);
                MMM_BLAS(B, &(G.updiag[i]), Guf1, Guf);
                matK(B, Guf, -1, &(G.updiag[i]));

                free(Guf), free(Guf1);
            }
        }

        MMM_BLAS(B, &(A.updiag[i]), Glf, Glf1);
        MMM_BLAS(B, &(G.mdiag[i]), Glf1, Glf);
        mat_ADD(B, &(G.mdiag[i]), Glf1, &(G.mdiag[i]));

        free(Glf), free(Glf1);
    }
}

int main() {
    float *t = new float[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = max(i - 1, 0); j < min(i + 2, 4); ++j) {
            t[4 * i + j] = 5;
        }
    }
    Matrix m(4, t), f(4);
    m.printM();
    m.DensetoB3D(1);
    f.DensetoB3D(1);
    // m.printB();
    rgf(m, f);
    // f.printB();
    f.B3DtoDense();
    f.printM();
    return 0;
}

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