## Steps to run on davinci
1. Try the folloing command to compile (the libraries installed on my home):
mpic++ -o test matrices_utils.cpp rgf1.cpp rgf2.cpp test.cpp argparse.cpp -I/home/bao_yifan/local/lapack/include -I/home/bao_yifan/local/openblas/include -L/home/bao_yifan/local/openblas/lib -L/home/bao_yifan/local/lapack/lib -lopenblas -llapack
2. If it does not work, copy the directories to your own home:
cp -r /home/bao_yifan/local ~/
3. use the command:
mpic++ -o test matrices_utils.cpp rgf1.cpp rgf2.cpp test.cpp argparse.cpp -I~/local/lapack/include -I~/local/openblas/include -L~/local/openblas/lib -L~/local/lapack/lib -lopenblas -llapack~
4. You can change Makefile accordingly to make it easy.

To include the paths on vscode:
1. go to the vscode cpp settings, add the path as mentioned in the previous command.

