## Steps to run on davinci
0. vim .bashrc, add the following into the bottom:
```shell
export LD_LIBRARY_PATH=/home/bao_yifan/local/lapack/lib:/home/bao_yifan/local/openblas/lib:$LD_LIBRARY_PATH
export BLAS=/home/bao_yifan/local/openblas/lib/libopenblas.a
export LAPACK=/home/bao_yifan/local/lapack/lib/liblapack.a
```
1. Try the folloing command to compile (the libraries installed on my home):
```shell
mpic++ -o test matrices_utils.cpp rgf1.cpp rgf2.cpp test.cpp argparse.cpp -I/home/bao_yifan/local/lapack/include -I/home/bao_yifan/local/openblas/include -L/home/bao_yifan/local/openblas/lib -L/home/bao_yifan/local/lapack/lib -lopenblas -llapack
```
2. If it does not work (it should work since if we have read access to others' files), copy the directories to your own home:
```shell
cp -r /home/bao_yifan/local ~/
```
Don't forget to change the .bashrc accordingly
3. use the command:
```shell
mpic++ -o test matrices_utils.cpp rgf1.cpp rgf2.cpp test.cpp argparse.cpp -I~/local/lapack/include -I~/local/openblas/include -L~/local/openblas/lib -L~/local/lapack/lib -lopenblas -llapack~
```
4. You can change Makefile accordingly to make it easy.

To include the paths on vscode:
1. go to the vscode cpp settings, add the path as mentioned in the previous command.

