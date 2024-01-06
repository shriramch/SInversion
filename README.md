# SInversion
In this project, we are implementing an optimized Recursive Green's Function (RGF) Selected Inversion Algorithm for inverting banded matrices. Matrices are stored in the Block-Tri-Diagonal format. There are two versions of the algorithm: ```rgf1sided``` and ```rgf2sided```. ```rgf2sided``` is a parallelized implementation of ```rgf1sided```, with two processes inverting two halves of the matrix and communicating through MPI. Each version also has an associated CUDA implementation optimized for NVIDIA GPU-based computation.

## Files
- ```rgf1.cpp```: Implementation of RGF1 algorithm.
- ```rgf2.cpp```: Implementation of RGF2 algorithm, with MPI.
- ```rgf1_cuda.cu```: Implementation of RGF1 algorithm, with CUDA.
- ```rgf2_cuda.cpp``` and ```cuda_kernels.cu```: Implementation of RGF2 algorithm, with MPI and CUDA.
- ```matrices_utils.cpp```: Implementation of matrix and utility routines
- ```test.cpp```: Benchmarking code, using the LibLSB library.
- ```test_correct.cpp```: Example file demonstrating the API and usage of the functions.
- ```benchmark.sh```: Benchmarking script that runs ```test.cpp``` on multiple input sizes.

## Dependencies
- BLAS library
- LAPACKe library 
- MPI
- GPU and CUDA support
- Cublas and Cusolver
- LibLSB Benchmarking Library
Modify the makefile to include the locations to the libraries.