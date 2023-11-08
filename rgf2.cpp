#include "rgf2.hpp"

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

void Matrix::convert3D(int blockSize) {
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

// References based implementation
// R = nblocks_2
void rgf2sided_upperprocess(Matrix &A, Matrix &G, int R, bool sym_mat = false,
                            bool save_off_diag = true) {}

// L = nblocks_2
void rgf2sided_lowerprocess(Matrix &A, Matrix &G, int L, bool sym_mat = false,
                            bool save_off_diag = true) {}

void rgf2sided(Matrix &A, Matrix &G, bool sym_mat = false,
               bool save_off_diag = true) {
    int process_rank, B, n;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    A.get(B, n);

    int nblocks_2 = B / 2;

    if (process_rank == 0) {
        rgf2sided_upperprocess(A, G, nblocks_2, sym_mat, save_off_diag);

        MPI_Recv((void *)G.mdiag, nblocks_2 * n * n, MPI_FLOAT, 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv((void *)G.updiag, nblocks_2 * n * n, MPI_FLOAT, 1, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv((void *)G.lodiag, nblocks_2 * n * n, MPI_FLOAT, 1, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    } else if (process_rank == 1) {
        rgf2sided_lowerprocess(A, G, nblocks_2, sym_mat, save_off_diag);

        MPI_Send((const void *)&G.mdiag[nblocks_2 * B * B],
                 (B - nblocks_2) * n * n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        MPI_Send((const void *)&G.updiag[nblocks_2 * B * B],
                 (B - nblocks_2 - 1) * n * n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        MPI_Send((const void *)&G.mdiag[nblocks_2 * B * B],
                 (B - nblocks_2 - 1) * n * n, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
    }
}

int main() {
    float *t = new float[16];
    for (int i = 0; i < 16; ++i) {
        t[i] = i + 1;
    }
    Matrix m(4, t);
    // m.printM();
    m.convert3D(1);
    // m.printB();
}

void Matrix::MMM_BLAS(int n, float *A, float *B, float *result) {
    const CBLAS_ORDER order = CblasRowMajor;
    const CBLAS_TRANSPOSE transA = CblasNoTrans;
    const CBLAS_TRANSPOSE transB = CblasNoTrans;
    const float alpha = 1.0;
    const float beta = 0.0;

    cblas_sgemm(order, transA, transB, n, n, n, alpha, A, n, B, n, beta, result,
                n);
}

void Matrix::MMM_noob(int n, float *A, float *B, float *result) {

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

void Matrix::mat_INV(int n, const float *A, float *result) {

    int *ipiv = (int *)malloc(n * sizeof(int));
    memcpy(result, A, n * n * sizeof(float));

    LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, result, n, ipiv);

    LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, result, n, ipiv);

    free(ipiv);
}
