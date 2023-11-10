#include <iostream>
#include <cblas.h>
using namespace std;
int main() {
    int blockSize = 4;
    float *A = new float[blockSize*blockSize]();
    float *B = new float[blockSize*blockSize]();
    for (int i = 0; i < blockSize * blockSize; i++) {
        A[i] = i;
    }
    cblas_somatcopy(CblasRowMajor, CblasTrans, blockSize, blockSize, 1.0f, A , blockSize, B, blockSize); 
    cout << "Matrix: " << endl;
    for (int i = 0; i < blockSize; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            cout << A[i * blockSize + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    for (int i = 0; i < blockSize; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            cout << B[i * blockSize + j] << " ";
        }
        cout << endl;
    }
}
