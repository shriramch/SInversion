#pragma once
#include <bits/stdc++.h>
#include <cblas.h>
#include <lapacke.h>

using namespace std;

class Matrix {
    int n, B;
    float *mat;

public:
    float *mdiag, *updiag, *lodiag;
    Matrix(int);
    Matrix(int, float *);
    Matrix(float *, float *, float *);
    void DensetoB3D(int);
    void B3DtoDense();
    void printM();
    void printB();
    void get(int &, int &);
};

void MMM_BLAS(int, float *, float *, float *);
void MMM_noob(int, float *, float *, float *);
void mat_INV(int, const float *, float *);
void mat_SUB(int, float *, float *, float *);
void mat_ADD(int, float *, float *, float *);
void matK(int, float *, int, float *);
void matT(int, float *, float *);

void rgf(Matrix &, Matrix &, bool, bool);
