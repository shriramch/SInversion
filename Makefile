UNAME := $(shell uname)
HOSTNAME := $(shell hostname)

CUDA := nvcc
CUDA_FLAGS := -lcublas -lcusolver

MPI := mpic++
MPI_FLAGS_DAVINCI := -lopenblas -llapack -I/home/bao_yifan/local/lapack/include -I/home/bao_yifan/local/openblas/include -L/home/bao_yifan/local/openblas/lib -L/home/bao_yifan/local/lapack/lib
MPI_FLAGS_MACOS := -I/usr/local/opt/lapack/include -I/usr/local/opt/openblas/include -L/usr/local/opt/openblas/lib -L/usr/local/opt/lapack/lib -lblas -llapack -llapacke
MPI_FLAGS_OTHERS := -lblas -llapack -llapacke

MPI_CUDA_LINK_FLAGS_DAVINCI := -I/usr/local/cuda-11.7/include -L/usr/local/cuda-11.7/lib64 -lcudart
LIBLSB_library := -I/home/akash_sood/liblsb/lib/include -L/home/akash_sood/liblsb/lib/lib -llsb 

print: run
	python3 parse_output.py

# C++ RGF1 and RGF2

run: compile_mpi
	mpirun -np 2 ./test -m 1024 -b 64 -n 10 -s 0 -o 1

ifeq ($(UNAME), Darwin)
compile_mpi:
	$(MPI) $(MPI_FLAGS_MACOS) -o test matrices_utils.cpp rgf1.cpp rgf2.cpp test.cpp argparse.cpp
else ifeq ($(HOSTNAME), davinci)
compile_mpi: 
	$(MPI) $(MPI_FLAGS_DAVINCI) -o test matrices_utils.cpp rgf1.cpp rgf2.cpp test.cpp argparse.cpp
else
compile_mpi:
	$(MPI) $(MPI_FLAGS_OTHERS) -o test matrices_utils.cpp rgf1.cpp rgf2.cpp test.cpp argparse.cpp
endif


# CUDA RGF1 and RGF2 (only davinci)

run_cuda1: compile_cuda1
	./test_cuda -m 8 -b 2 -n 10 -s 0 -o 1

compile_cuda1:
	$(CUDA) $(MPI_FLAGS_DAVINCI) $(CUDA_FLAGS) -o test_cuda rgf1_cuda.cu matrices_utils.cpp rgf1.cpp argparse.cpp

run_cuda2: compile_cuda2
	mpirun -np 2 ./test_cuda -m 8 -b 2 -n 10 -s 0 -o 1

compile_cuda2:
	$(MPI) -c $(MPI_FLAGS_DAVINCI) $(MPI_CUDA_LINK_FLAGS_DAVINCI) $(CUDA_FLAGS) rgf2_cuda.cpp rgf2.cpp matrices_utils.cpp argparse.cpp
	$(CUDA) -c $(CUDA_FLAGS) cuda_kernels.cu
	$(MPI) *.o $(MPI_FLAGS_DAVINCI) $(MPI_CUDA_LINK_FLAGS_DAVINCI) $(CUDA_FLAGS) -o test_cuda
	rm *.o

# Benchmarking (only C++)

clean:
	rm -fR main plot_out/* run.txt

CPP1 = -DENABLE_LIBLSB1
CPP2 = -DENABLE_LIBLSB2
CUDA1 = -DENABLE_LIBLSB_C1
CUDA2 = -DENABLE_LIBLSB_C2

compile_rgf1:
	$(MPI) -o test rgf1.cpp matrices_utils.cpp test.cpp argparse.cpp $(CPP1) $(MPI_FLAGS_DAVINCI) $(LIBLSB_library) 

lsb1: compile_rgf1
	mpirun -np 1 ./test -m 16 -b 4 -n 10 -s 0 -o 1 > run.txt

compile_rgf2:
	$(MPI) -o test rgf2.cpp matrices_utils.cpp test.cpp argparse.cpp $(CPP2) $(MPI_FLAGS_DAVINCI) $(LIBLSB_library)

lsb2: compile_rgf2
	mpirun -np 2 ./test -m 16384 -b 256 -n 10 -s 0 -o 1 > run.txt

compile_cuda_rgf1:
	$(MPI) -c $(MPI_FLAGS_DAVINCI) $(MPI_CUDA_LINK_FLAGS_DAVINCI) $(CUDA_FLAGS) $(LIBLSB_library) $(CUDA1) test.cpp matrices_utils.cpp argparse.cpp
	$(CUDA) -c $(CUDA_FLAGS) rgf1_cuda.cu
	$(MPI) *.o $(MPI_FLAGS_DAVINCI) $(MPI_CUDA_LINK_FLAGS_DAVINCI) $(CUDA_FLAGS) $(LIBLSB_library) -o test_cuda
	rm *.o

lsbc1: compile_cuda_rgf1
	mpirun -np 1 ./test_cuda -m 64 -b 8 -n 10 -s 0 -o 1 > run.txt

compile_cuda_rgf2:
	$(MPI) -c $(MPI_FLAGS_DAVINCI) $(MPI_CUDA_LINK_FLAGS_DAVINCI) $(CUDA_FLAGS) $(LIBLSB_library) $(CUDA2) rgf2_cuda.cpp test.cpp matrices_utils.cpp argparse.cpp
	$(CUDA) -c $(CUDA_FLAGS) cuda_kernels.cu
	$(MPI) *.o $(MPI_FLAGS_DAVINCI) $(MPI_CUDA_LINK_FLAGS_DAVINCI) $(CUDA_FLAGS) $(LIBLSB_library) -o test_cuda
	rm *.o

lsbc2: compile_cuda_rgf2
	mpirun -np 2 ./test_cuda -m 128 -b 16 -n 10 -s 0 -o 1 > run.txt

