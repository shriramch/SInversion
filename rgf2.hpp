#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

class Matrix {
    int n, B;
    float *mat;
    vector<float *> mdiag, updiag, lodiag;

public:
    Matrix(int);
    Matrix(int, float *);
    void convert3D(int);
    void printM();
    void printB();
};

void MMM(float *A, float *B, float *C);
void MI(float *A, float *Ai);