UNAME := $(shell uname)
MPI := mpic++
MPI_FLAGS_MACOS := -I/opt/homebrew/opt/lapack/include -I/opt/homebrew/opt/openblas/include -L/opt/homebrew/opt/openblas/lib -L/opt/homebrew/opt/lapack/lib -lblas -llapack -llapacke
MPI_FLAGS_OTHERS := -lblas -llapack -llapacke
CUDA := nvcc

print: run
	python3 parse_output.py

run: compile_mpi
	mpirun -np 2 ./test -m 8 -b 2 -n 10 -s 0 -o 1  > run.txt

ifeq ($(UNAME), Darwin)
compile_mpi:
	$(MPI) -o test matrices_utils.cpp rgf1.cpp rgf2.cpp test.cpp argparse.cpp $(MPI_FLAGS_MACOS)
else
compile_mpi:
	$(MPI) -o test matrices_utils.cpp rgf1.cpp rgf2.cpp test.cpp argparse.cpp $(MPI_FLAGS_OTHERS)
endif

clean:
	rm -fR main plot_out/* run.txt


run_cuda: compile_cuda
	./test_cuda

compile_cuda:
	$(CUDA) -o test_cuda rgf1_cuda.cu

