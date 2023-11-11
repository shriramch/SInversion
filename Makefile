MPI := mpic++
MPI_FLAGS := -I/usr/local/opt/lapack/include -I/usr/local/opt/openblas/include  -L/usr/local/opt/openblas/lib -L/usr/local/opt/lapack/lib -lblas -llapack -llapacke
#-Werror , and added -w to avoid warning (have some problem with %lu as I's using Windows)

print: run
	python3 parse_output.py

run: compile_mpi
	mpirun -np 2 ./test > run.txt

compile_mpi:
	$(MPI) -o test rgf2.cpp test.cpp matrices_utils.cpp $(MPI_FLAGS) 
# TODO, reorginize to put code into 2 folders 'source' and 'include'

clean:
	rm -fR main plot_out/* run.txt