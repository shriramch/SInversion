UNAME := $(shell uname)
HOSTNAME := $(shell hostname)
MPI := mpic++
MPI_FLAGS_MACOS := -I/usr/local/opt/lapack/include -I/usr/local/opt/openblas/include -L/usr/local/opt/openblas/lib -L/usr/local/opt/lapack/lib -lblas -llapack -llapacke
MPI_FLAGS_OTHERS := -lblas -llapack -llapacke
MPI_FLAGS_DAVINCI := -I/home/bao_yifan/local/lapack/include -I/home/bao_yifan/local/openblas/include -L/home/bao_yifan/local/openblas/lib -L/home/bao_yifan/local/lapack/lib -lopenblas -llapack

print: run
	python3 parse_output.py

run: compile_mpi
	mpirun -np 2 ./test -m 8 -b 2 -n 10 -s 0 -o 1

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

# Keeping libLSB module seprate for now
CXXFLAGS += -DENABLE_LIBLSB
compile_lsb:
	$(MPI) -o test *.cpp $(CXXFLAGS) $(MPI_FLAGS_OTHERS) -llsb

rgf1_test: compile_lsb
	mpirun -np 1 ./test -m 8 -b 2 -n 10 -s 0 -o 1  > run.txt

clean:
	rm -fR main plot_out/* run.txt
