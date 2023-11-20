#include "rgf1.hpp"
#include <fstream>
#include <iostream>
#include "liblsb.h"

using namespace std;

int main() {

    LSB_Init("DPHPC Project", 0);
    fstream fin("test.txt", ios::in);

    int n, blocksize;
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '=');
    fin >> n;
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '=');
    fin >> blocksize;

    cout<<"n: "<<n<<" blocksize: "<<blocksize<<endl;


    float *temp_mat = new float[n*n];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fin >> temp_mat[i*n + j];
        }
    }

    //initialise input matrix
    Matrix A(n, temp_mat);

    //init mdiag, updiag, lodiag
    A.DensetoB3D(blocksize);

    //init result matrix
    Matrix G(n);
    G.DensetoB3D(blocksize);

    int RUNS= 10;
    for (int i=0; i<RUNS; i++){
        //reset 
        LSB_Res();

        //main algo
        rgf(A, G, false, true);

        //end benchmarking
        LSB_Rec(i);
    }

    //converting to normal format
    G.B3DtoDense();

    G.printB();


    LSB_Finalize();
    return 0;
}