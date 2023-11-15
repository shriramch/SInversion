#include "rgf1.hpp"
#include <fstream>
#include <iostream>

using namespace std;

int main() {
    fstream fin("test.txt", ios::in);
    float *m = new float[25];
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            fin >> m[5 * i + j];
        }
    }
    // for (int i = 0; i < 5; ++i) {
    //     for (int j = 0; j < 5; ++j) {
    //         cout << m[5 * i + j] << " ";
    //     }
    //     cout << endl;
    // }

    Matrix A(5, m);
    A.DensetoB3D(1);
    Matrix G(5);
    G.DensetoB3D(1);

    rgf(A, G, true, true);

    G.B3DtoDense();

    G.printB();

    return 0;
}