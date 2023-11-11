UNAME := $(shell uname)
MPI := mpic++
MPI_FLAGS_MACOS := -I/usr/local/opt/lapack/include -I/usr/local/opt/openblas/include -L/usr/local/opt/openblas/lib -L/usr/local/opt/lapack/lib -lblas -llapack -llapacke
MPI_FLAGS_OTHERS := -lblas -llapack -llapacke

print: run
	python3 parse_output.py

run: compile_mpi
	mpirun -np 2 ./test -m 8 -b 2 -n 10 -s 0 -o 1  > run.txt

ifeq ($(UNAME), Darwin)
compile_mpi:
	$(MPI) -o test matrices_utils.cpp rgf2.cpp test.cpp argparse.cpp $(MPI_FLAGS_MACOS)
else
compile_mpi:
	$(MPI) -o test matrices_utils.cpp rgf2.cpp test.cpp argparse.cpp $(MPI_FLAGS_OTHERS)
endif

clean:
	rm -fR main plot_out/* run.txt
