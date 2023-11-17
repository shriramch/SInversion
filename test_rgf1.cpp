#include "rgf1.hpp"
#include <fstream>
#include <iostream>

using namespace std;

int main() {
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

    //main algo
    rgf(A, G, false, true);

    //converting to normal format
    G.B3DtoDense();

    G.printB();

    return 0;
}