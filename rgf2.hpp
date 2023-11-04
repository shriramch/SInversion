#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

class Matrix {
    int n, B;
    float *mat;

public:
    float *mdiag, *updiag, *lodiag;
    Matrix(int);
    Matrix(int, float *);
    Matrix(float *, float *, float *);
    void convert3D(int);
    void printM();
    void printB();
    void get(int &, int &);
};

void MMM(float *A, float *B, float *C);
void MI(float *A, float *Ai);