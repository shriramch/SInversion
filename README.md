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
- (random mode): mpirun -np 2 ./test -m <matrixSize> -b <blockSize=2> -n <numRuns=10> -s <isSymmetric=false> -o <saveOffDiag=true>\n . Eg. mpirun -np 2 ./test -m 8 -b 2 -n 10 -s 0 -o 1
- (file mode): mpirun -np 2 ./test -m <matrixSize> -b <blockSize=2> -n <numRuns=10> -s <isSymmetric=false> -o <saveOffDiag=true> -f <inputPath>\n

## Benchmarking
- ./benchmark.sh Runs the benchmarking script
- Run nohup ./benchmark.sh > output.log 2>&1 & to run it in the background

Very easy use:
1. Implement your algorithm
2. Include it on test.cpp
3. Wrap it as rgf2sidedAlgorithm
4. Put it on algorithms vectors
5. make run 
6. see the result on run.txt
   

TODO:
- Inspect memory safety issues.
- See if precision can be improved.
- Integrate and profiling.

Note:
- Please do not use using namespace std;
- Please do not use #include <bits/stdc++.h>
- mpic++ -o test rgf2.cpp -lblas -llapack

Problems:
- on Davinci cannot run make as cblas.h is not there, and no sudo permission to install:
  - [BLAS](#blas) :   sudo apt install libopenblas-base 
                      sudo apt-get install libopenblas-dev (Ubuntu)
  - [LAPACKe](#lapacke) : sudo apt-get install liblapacke-dev
  - [MPI](#mpi) : sudo apt-get install openmpi-bin libopenmpi-dev

TODO, after Andrea:
  - ask Permission for SUDO or at least figure it out how to install the above
  - test the following functions: (added blindly without testing)
    - matrixInversionKernel
    - matrixTransposeKernel
  - test the rgf1sided_cuda  -> note that for now I did an 1v1 translation from the cpp implementation, more room for improvement

Next steps:
1. Mid-term Meeting - 24 November
   - CUDA Implementation RGF1Sided
     + Setting up the CUDA Environment
     + Matrix Representation: format suitable for GPU memory -> CSR (Compressed Sparse Row) or COO (Coordinate List) 
     + Memory Management: memory allocation on the GPU. transfer from the host (CPU) to the device (GPU).
     + CUDA kernels for the operations including matrix addition/substraction, inversions, and multiplications.
     + Define Parallelization Strategy
     + Testing and Validation
       
   - Benchmark Implementation
     + Define Benchmark KPI's / Measurements: Execution Time, Scalability, #Flops, Flops/s
     + Implementation for RGF1sided and RGF2sided (leverage LibSciBench? - https://spcl.inf.ethz.ch/Research/Performance/LibLSB/)
     + Setup for Benchmarking / Environment Configuration on Darwin
     + Statistical Data Collection (ensuring "Warm Registers") incl. Multiple Runs and Analysis (e.g., leavering Seaborn / Python or R)
    
   - Alignment on Thursday, 23 November
      + consolidate status quo and results
      + collect open question for mid-term review
        
2. Mid of December
   - Cuda Implementation RGF2Sided (MPI + CUDA)
   - Improve Performance leverag Streaming RGF1Sided (CUDA) & RGF2Sided (MPI + CUDA)
   - Analyis & Conclusion - first draft of Report

do the profiling to get flops
example:
```
mpirun -np 1 ncu --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum ./test -m 16 -b 4 -n 10 -s 0 -o 1 > flops_16_rgf1.txt 

Reference Nvidia document:
https://docs.nvidia.com/nsight-compute/pdf/NsightComputeCli.pdf (page 36)

nvprov --metrix flop_count_sp, flop_count_dp didn't work on davinci

``` 

