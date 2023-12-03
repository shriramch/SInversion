UNAME := $(shell uname)
HOSTNAME := $(shell hostname)
MPI := mpic++
MPI_FLAGS_MACOS := -I/usr/local/opt/lapack/include -I/usr/local/opt/openblas/include -L/usr/local/opt/openblas/lib -L/usr/local/opt/lapack/lib -lblas -llapack -llapacke
MPI_FLAGS_OTHERS := -lblas -llapack -llapacke

print: run
	python3 parse_output.py

run: compile_mpi
	mpirun -np 2 ./test -m 1024 -b 64 -n 10 -s 0 -o 1

ifeq ($(UNAME), Darwin)
compile_mpi:
	$(MPI) -o test matrices_utils.cpp rgf1.cpp rgf2.cpp test.cpp argparse.cpp $(MPI_FLAGS_MACOS)
else
compile_mpi:
	$(MPI) -o test matrices_utils.cpp rgf1.cpp rgf2.cpp test.cpp argparse.cpp $(MPI_FLAGS_OTHERS)
endif

clean:
	rm -fR main plot_out/* run.txt


##############################################################################################################
CXXFLAGS1 = -DENABLE_LIBLSB1
CXXFLAGS2 = -DENABLE_LIBLSB2

compile_rgf1:
	$(MPI) -o test *.cpp $(CXXFLAGS1) $(MPI_FLAGS_OTHERS) -llsb

lsb1: compile_rgf1
	mpirun -np 1 ./test -m 256 -b 16 -n 10 -s 0 -o 1 > run.txt

compile_rgf2:
	$(MPI) -o test *.cpp $(CXXFLAGS2) $(MPI_FLAGS_OTHERS) -llsb

lsb2: compile_rgf2
	mpirun -np 2 ./test -m 256 -b 16 -n 10 -s 0 -o 1 > run.txt

###############################################################################################################
MPI_FLAGS_DAVINCI := -I/home/bao_yifan/local/lapack/include -I/home/bao_yifan/local/openblas/include -L/home/bao_yifan/local/openblas/lib -L/home/bao_yifan/local/lapack/lib -lopenblas -llapack
CUDA := nvcc
CUDA_FLAGS_OTHERS := -lcublas -lcusolver

run_cuda: compile_cuda
	./test_cuda -m 8 -b 2 -n 10 -s 0 -o 1

compile_cuda:
# $(CUDA) -o test_cuda rgf1_cuda.cu matrices_utils.cpp $(CUDA_FLAGS_OTHERS) $(MPI_FLAGS_DAVINCI)
# $(CUDA) -o test_cuda temp.cu  $(CUDA_FLAGS_OTHERS) $(MPI_FLAGS_DAVINCI)
# $(CUDA) -o test_cuda cusolver_getrf_example.cu  $(CUDA_FLAGS_OTHERS) $(MPI_FLAGS_DAVINCI)
	$(CUDA) -o test_cuda rgf1_cuda.cu matrices_utils.cpp rgf1.cpp argparse.cpp $(MPI_FLAGS_DAVINCI) $(CUDA_FLAGS_OTHERS) $(MPI_FLAGS_DAVINCI)
