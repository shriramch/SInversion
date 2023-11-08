#ifndef RGF
#define RGF
#include <bits/stdc++.h>
#include <cblas.h>
#include <lapacke.h>
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
    void MMM_BLAS(int, float *, float *, float *);
    void MMM_noob(int, float *, float *, float *);
    void mat_INV(int n, const float *A, float *result);
};

#endif