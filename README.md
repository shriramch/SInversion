# Project README

This project is designed to implement a optimised code for Selected Inversion Algorithm.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- [BLAS](#blas) : sudo apt install libopenblas-base 
- [LAPACKe](#lapacke) : sudo apt-get install liblapacke-dev
- [MPI](#mpi) : sudo apt-get install openmpi-bin libopenmpi-dev

## Compilation Instructions

- mpic++ -o test rgf2.cpp -lblas -llapack