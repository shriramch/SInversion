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
else ifeq ($(HOSTNAME), davinci)
compile_mpi: 
	$(MPI) -o test matrices_utils.cpp rgf1.cpp rgf2.cpp test.cpp argparse.cpp $(MPI_FLAGS_DAVINCI)
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
# Open 4.1.6
MPI_FLAGS_DAVINCI := -lopenblas -llapack -I/home/bao_yifan/local/lapack/include -I/home/bao_yifan/local/openblas/include -L/home/bao_yifan/local/openblas/lib -L/home/bao_yifan/local/lapack/lib
# -lmpi_cxx
# -I/usr/include/x86_64-linux-gnu/mpich
CUDA := nvcc
CUDA_FLAGS_OTHERS := -lcublas -lcusolver
# MPI_FLAGS_DAVINCI :=  -lopenblas -llapack -I/home/bao_yifan/local/lapack/include -I/home/bao_yifan/local/openblas/include -L/home/bao_yifan/local/openblas/lib -L/home/bao_yifan/local/lapack/lib
# MPI_FLAGS_DAVINCI += $(shell mpicc -show)
MPI_CUDA_LINK_FLAGS := -I/usr/local/cuda-11.7/include -L/usr/local/cuda-11.7/lib64 -lcudart

run_cuda: compile_cuda
	./test_cuda -m 8 -b 2 -n 10 -s 0 -o 1

compile_cuda:
# $(CUDA) -o test_cuda rgf1_cuda.cu matrices_utils.cpp $(CUDA_FLAGS_OTHERS) $(MPI_FLAGS_DAVINCI)
# $(CUDA) -o test_cuda temp.cu  $(CUDA_FLAGS_OTHERS) $(MPI_FLAGS_DAVINCI)
# $(CUDA) -o test_cuda cusolver_getrf_example.cu  $(CUDA_FLAGS_OTHERS) $(MPI_FLAGS_DAVINCI)
# $(CUDA) -o test_cuda rgf1_cuda.cu matrices_utils.cpp rgf1.cpp argparse.cpp $(MPI_FLAGS_DAVINCI) $(CUDA_FLAGS_OTHERS)
	$(CUDA) -o test_cuda rgf2_cuda.cu matrices_utils.cpp rgf2.cpp argparse.cpp $(MPI_FLAGS_DAVINCI) $(CUDA_FLAGS_OTHERS)

run_cuda2: compile_cuda2
	mpirun -np 2 ./test_cuda -m 8 -b 2 -n 10 -s 0 -o 1

compile_cuda2:
# $(CUDA) -c rgf2_cuda.cu matrices_utils.cpp rgf2.cpp argparse.cpp $(MPI_FLAGS_DAVINCI) $(CUDA_FLAGS_OTHERS)
# $(CUDA) -lm -lcudart $(CUDA_FLAGS_OTHERS) $(MPI_FLAGS_DAVINCI) -lmpi_cxx -lopen-rte -lopen-pal -ldl -lnsl -lutil -lm *.o -o test_cuda
# rm *.o
	$(MPI) -c rgf2_cuda.cpp rgf2.cpp matrices_utils.cpp $(MPI_FLAGS_DAVINCI) $(MPI_CUDA_LINK_FLAGS) $(CUDA_FLAGS_OTHERS)
	$(CUDA) -c rgf2_cuda_kernels.cu $(CUDA_FLAGS_OTHERS)
	$(MPI) *.o $(MPI_FLAGS_DAVINCI) $(MPI_CUDA_LINK_FLAGS) $(CUDA_FLAGS_OTHERS) -o test_cuda
	rm *.o