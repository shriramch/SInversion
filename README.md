# Project README

This project is designed to implement a optimised code for Selected Inversion Algorithm.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- sudo apt-get upgrade
- [BLAS](#blas) :   sudo apt install libopenblas-base 
                    sudo apt-get install libopenblas-dev (Ubuntu)
- [LAPACKe](#lapacke) : sudo apt-get install liblapacke-dev
- [MPI](#mpi) : sudo apt-get install openmpi-bin libopenmpi-dev

## Compilation Instructions

- mpic++ -o test rgf2.cpp -lblas -llapack 

For Ubuntu user:
- mpic++ -o test rgf2.cpp -lblas -llapacke
- mpic++ -o test rgf2.cpp test.cpp matrices_utils.cpp -lblas -llapacke


for MacOS user (add path to openblas and lapack libraries):
- mpic++ -o test rgf2.cpp -I/usr/local/opt/lapack/include -I/usr/local/opt/openblas/include  -L/usr/local/opt/openblas/lib -L/usr/local/opt/lapack/lib -lopenblas -llapack

detailed test (buggy, no try, use only the rgf2.cpp main to test now)
- mpic++ -o test matrices_utils.cpp test.cpp rgf2.cpp -I/usr/local/opt/lapack/include -I/usr/local/opt/openblas/include  -L/usr/local/opt/openblas/lib -L/usr/local/opt/lapack/lib -lopenblas -llapack

## Running Instructions
run with two processes
-  mpirun -np 2 ./test


## How to Test
- 'make run', it will compile the files and then save the output to 'run.txt'

The new running method:
- (random mode): mpirun -np 2 ./test -m <matrixSize> -b <blockSize=2> -n <numRuns=10> -s <isSymmetric=false> -o <saveOffDiag=true>\n 
- (file mode): mpirun -np 2 ./test -m <matrixSize> -b <blockSize=2> -n <numRuns=10> -s <isSymmetric=false> -o <saveOffDiag=true> -f <inputPath>\n
        
TODO:
- Inspect memory safety issues
- Integrate and profiling.