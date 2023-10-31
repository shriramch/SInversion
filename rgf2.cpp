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

void Matrix::convert3D(int blockSize) {
    B = blockSize;
    assert(n % blockSize == 0);
    int numBlocks = n / blockSize;
    mdiag.resize(numBlocks);
    updiag.resize(numBlocks - 1);
    lodiag.resize(numBlocks - 1);
    for (int b = 0; b < numBlocks; ++b) {
        mdiag[b] = new float[blockSize * blockSize];
        for (int i = 0; i < blockSize; ++i) {
            for (int j = 0; j < blockSize; ++j) {
                mdiag[b][i * blockSize + j] =
                    mat[(b * blockSize + i) * n + (b * blockSize + j)];
            }
        }
    }

    for (int b = 0; b < numBlocks - 1; ++b) {
        updiag[b] = new float[blockSize * blockSize];
        lodiag[b] = new float[blockSize * blockSize];
        for (int i = 0; i < blockSize; ++i) {
            for (int j = 0; j < blockSize; ++j) {
                updiag[b][i * blockSize + j] =
                    mat[(b * blockSize + i) * n + ((b + 1) * blockSize + j)];
                lodiag[b][i * blockSize + j] =
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
                cout << mdiag[i][j * B + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << "Upper diagonal: " << endl;
    for (int i = 0; i < n / B - 1; ++i) {
        for (int j = 0; j < B; ++j) {
            for (int k = 0; k < B; ++k) {
                cout << updiag[i][j * B + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << "Lower diagonal: " << endl;
    for (int i = 0; i < n / B - 1; ++i) {
        for (int j = 0; j < B; ++j) {
            for (int k = 0; k < B; ++k) {
                cout << lodiag[i][j * B + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

// int main() {
//     float *t = new float[16];
//     for (int i = 0; i < 16; ++i) {
//         t[i] = i + 1;
//     }
//     Matrix m(4, t);
//     m.printM();
//     m.convert3D(1);
//     m.printB();
// }
