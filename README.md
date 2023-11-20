# Project README

This project is designed to implement a optimised code for Selected Inversion Algorithm.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- [BLAS](#blas) : sudo apt install libopenblas-base 
- [LAPACKe](#lapacke) : sudo apt-get install liblapacke-dev
- [MPI](#mpi) : sudo apt-get install openmpi-bin libopenmpi-dev

## Compilation Instructions

- g++ -o test test_rgf1.cpp rgf1.cpp -lblas -llapacke -llsb

## For Testing

- run python program defining no_block and blocksize 
- python program will generate matrix and run the python version of rgf
- run the cpp code that will use the generated matrix from test.txt
- result of cpp | python program should be almost same (ignoring precision error)