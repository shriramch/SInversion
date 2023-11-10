MPI := mpic++
MPI_FLAGS := -lblas -llapack 
#-Werror , and added -w to avoid warning (have some problem with %lu as I's using Windows)
INCLUDES := -I include -lm

print: run
	python3 parse_output.py

run: compile_mpi
	./main > run.txt

compile_mpi:
	$(MPI) $(FLAGS) -o main src/* $(INCLUDES)

clean:
	rm -fR main plot_out/* run.txt