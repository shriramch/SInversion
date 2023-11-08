# Project README

This project is designed to implement a optimised code for Selected Inversion Algorithm.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- [BLAS](#blas) : sudo apt install libopenblas-base 
- [LAPACKe](#lapacke) : sudo apt-get install liblapacke-dev
- [MPI](#mpi) : sudo apt-get install openmpi-bin libopenmpi-dev

## Compilation Instructions

- mpic++ -o test rgf2.cpp -lblas -llapack

for MacOS user (add path to openblas and lapack libraries):
- mpic++ -o test rgf2.cpp -I/usr/local/opt/lapack/include -I/usr/local/opt/openblas/include  -L/usr/local/opt/openblas/lib -L/usr/local/opt/lapack/lib -lopenblas -llapack

detailed test (buggy, no try, use only the rgf2.cpp main to test now)
- mpic++ -o test matrices_utils.cpp test.cpp rgf2.cpp -I/usr/local/opt/lapack/include -I/usr/local/opt/openblas/include  -L/usr/local/opt/openblas/lib -L/usr/local/opt/lapack/lib -lopenblas -llapack

run with two processes
-  mpirun -np 2 ./test


TODO:
- write test code similar to testpy
- debug implementation