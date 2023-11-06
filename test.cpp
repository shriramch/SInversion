#include "rgf2.hpp"

int main() {
    float *t = new float[16];
    for (int i = 0; i < 16; ++i) {
        t[i] = i + 1;
    }
    Matrix A(4, t);
    A.printM();
    A.convert3D(2);
    A.printB();

    Matrix G(4); // zero initialization, same shape as A
    G.convert3D(2); // G has same blockSize as in A
    // rgf2sided(A, G,false, false);
}