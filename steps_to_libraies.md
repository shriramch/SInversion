## Steps to run on davinci
0. vim .bashrc, add the following into the bottom:
```shell
export LD_LIBRARY_PATH=/home/bao_yifan/local/lapack/lib:/home/bao_yifan/local/openblas/lib:$LD_LIBRARY_PATH
export CPATH=/home/bao_yifan/local/openblas/include:/home/bao_yifan/local/lapack/include
export OPENBLAS=/home/bao_yifan/local/openblas/lib/libopenblas.a
export LAPACK=/home/bao_yifan/local/lapack/lib/liblapack.a
```
source ~/.bashrc
1. Try the folloing command to compile (the libraries installed on my home):
```shell
mpic++ -o test matrices_utils.cpp rgf1.cpp rgf2.cpp test.cpp argparse.cpp -I/home/bao_yifan/local/lapack/include -I/home/bao_yifan/local/openblas/include -L/home/bao_yifan/local/openblas/lib -L/home/bao_yifan/local/lapack/lib -lopenblas -llapack
```
or you can use mpic++ -o test matrices_utils.cpp rgf1.cpp rgf2.cpp test.cpp argparse.cpp $OPENBLAS $LAPACK

2. If it does not work (it should work since if we have read access to others' files), copy the directories to your own home:
```shell
cp -r /home/bao_yifan/local ~/
```
Don't forget to change the .bashrc accordingly
3. use the command:
```shell
mpic++ -o test matrices_utils.cpp rgf1.cpp rgf2.cpp test.cpp argparse.cpp -I$HOME/local/lapack/include -I$HOME/local/openblas/include -L$HOME/local/openblas/lib -L$HOME/local/lapack/lib -lopenblas -llapack
```
4. Makefiles changed. Use make run is ok.


To include the paths on vscode:
1. go to the vscode cpp settings, add the path as mentioned in the previous command.

