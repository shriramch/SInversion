#include "rgf1.hpp"
#include "matrices_utils.hpp"
#include <cassert>


// Print the matrix
void printMatrix(const float* matrix, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

void rgf1sided(Matrix &A, Matrix &G, bool sym_mat = false,
               bool save_off_diag = true) {
    int blockSize, matrixSize;
    A.getBlockSizeAndMatrixSize(blockSize, matrixSize);
    // 0. Inverse of the first block
    A.invBLAS(blockSize, A.mdiag, G.mdiag);

    int nblocks = matrixSize / blockSize;

    // 1. Forward substitution (performed left to right)

    
    // std::cout << "CPP Before 1. Forward substitution (performed left to right)\n";
    // std::cout << "printing A_updiag from CUDA: \n"; 
    // printMatrix(A.updiag, blockSize , blockSize);
    // std::cout << "printing A_mdiag from CUDA: \n"; 
    // printMatrix(A.mdiag, blockSize , blockSize);
    // printMatrix(&(A.mdiag[blockSize*blockSize]), blockSize , blockSize);
    // std::cout << "printing A_lodiag from CUDA: \n"; 
    // printMatrix(A.lodiag, blockSize , blockSize);;
    // std::cout << "printing G_mdiag from CUDA: \n"; 
    // printMatrix(G.mdiag, blockSize , blockSize);
    // printMatrix(&(G.mdiag[blockSize*blockSize]), blockSize , blockSize);
    // std::cout << "----------------------------------------------------------------------- \n"; 

    for (int i = 1; i < nblocks; ++i) {
        std::cout << "CPP BEFORE " << i << ". INSIDE FIRST loop (performed left to right)\n";
        std::cout << "printing A_updiag from CUDA: \n"; 
        printMatrix(A.updiag, blockSize , blockSize);
        std::cout << "printing A_mdiag from CUDA: \n"; 
        printMatrix(A.mdiag, blockSize , blockSize);
        printMatrix(&(A.mdiag[blockSize*blockSize]), blockSize , blockSize);
        std::cout << "printing A_lodiag from CUDA: \n"; 
        printMatrix(A.lodiag, blockSize , blockSize);;
        std::cout << "printing G_mdiag from CUDA: \n"; 
        printMatrix(G.mdiag, blockSize , blockSize);
        printMatrix(&(G.mdiag[blockSize*blockSize]), blockSize , blockSize);
        std::cout << "----------------------------------------------------------------------- \n"; 
        
        float *AAi = new float[blockSize * blockSize](),
              *AGi = new float[blockSize * blockSize]();
        A.mmmBLAS(blockSize, &(A.lodiag[(i - 1) * blockSize * blockSize]),
                  &(G.mdiag[(i - 1) * blockSize * blockSize]), AGi);
        std::cout << "STEP BY STEP 1 printing AGi from CUDA: \n"; 
        printMatrix(AGi, blockSize , blockSize);

        A.mmmBLAS(blockSize, AGi, &(A.updiag[(i - 1) * blockSize * blockSize]),
                  AAi);
        std::cout << "STEP BY STEP 2 printing AAi from CUDA: \n"; 
        printMatrix(AAi, blockSize , blockSize);

        A.mmSub(blockSize, &(A.mdiag[i * blockSize * blockSize]), AAi, AGi);
        std::cout << "STEP BY STEP 3 printing AGi from CUDA: \n"; 
        printMatrix(AGi, blockSize , blockSize);

        A.invBLAS(blockSize, AGi, &(G.mdiag[i * blockSize * blockSize]));
        std::cout << "STEP BY STEP 4 printing AGi from CUDA: \n"; 
        printMatrix(&(G.mdiag[i * blockSize * blockSize]), blockSize , blockSize);

        delete[] AAi;
        delete[] AGi;
    }
    
    std::cout << "After 1. Forward substitution (performed left to right)\n";
    std::cout << "printing A_updiag from CUDA: \n"; 
    printMatrix(A.updiag, blockSize , blockSize);
    std::cout << "printing A_mdiag from CUDA: \n"; 
    printMatrix(A.mdiag, blockSize , blockSize);
    printMatrix(&(A.mdiag[blockSize*blockSize]), blockSize , blockSize);
    std::cout << "printing A_lodiag from CUDA: \n"; 
    printMatrix(A.lodiag, blockSize , blockSize);;
    std::cout << "printing G_mdiag from CUDA: \n"; 
    printMatrix(G.mdiag, blockSize , blockSize);
    printMatrix(&(G.mdiag[blockSize*blockSize]), blockSize , blockSize);
    std::cout << "----------------------------------------------------------------------- \n"; 


    for (int i = nblocks - 2; i >= 0; --i) {
        // std::cout << "CPP BEFORE " << i << ". INSIDE second loop (performed left to right)\n";
        // std::cout << "printing A_updiag from CUDA: \n"; 
        // printMatrix(A.updiag, blockSize , blockSize);
        // std::cout << "printing A_mdiag from CUDA: \n"; 
        // printMatrix(A.mdiag, blockSize , blockSize);
        // printMatrix(&(A.mdiag[blockSize*blockSize]), blockSize , blockSize);
        // std::cout << "printing A_lodiag from CUDA: \n"; 
        // printMatrix(A.lodiag, blockSize , blockSize);;
        // std::cout << "printing G_mdiag from CUDA: \n"; 
        // printMatrix(G.mdiag, blockSize , blockSize);
        // printMatrix(&(G.mdiag[blockSize*blockSize]), blockSize , blockSize);
        // std::cout << "----------------------------------------------------------------------- \n"; 
        
        float *Glf = new float[blockSize * blockSize](),
              *Glf1 = new float[blockSize * blockSize]();
        A.mmmBLAS(blockSize, &(G.mdiag[(i + 1) * blockSize * blockSize]),
                  &(A.lodiag[i * blockSize * blockSize]), Glf1);
        A.mmmBLAS(blockSize, Glf1, &(G.mdiag[i * blockSize * blockSize]), Glf);

        if (save_off_diag) {
            A.matScale(blockSize, Glf, -1,
                       &(G.lodiag[i * blockSize * blockSize]));
            if (sym_mat) {
                A.transposeBLAS(blockSize,
                                &(G.lodiag[i * blockSize * blockSize]),
                                &(G.updiag[i * blockSize * blockSize]));
            } else {
                float *Guf = new float[blockSize * blockSize](),
                      *Guf1 = new float[blockSize * blockSize]();
                A.mmmBLAS(blockSize, &(A.updiag[i * blockSize * blockSize]),
                          &(G.mdiag[(i + 1) * blockSize * blockSize]), Guf1);
                A.mmmBLAS(blockSize, &(G.mdiag[i * blockSize * blockSize]),
                          Guf1, Guf);
                A.matScale(blockSize, Guf, -1,
                           &(G.updiag[i * blockSize * blockSize]));

                delete[] Guf;
                delete[] Guf1;
            }
        }

        A.mmmBLAS(blockSize, &(A.updiag[i * blockSize * blockSize]), Glf, Glf1);
        A.mmmBLAS(blockSize, &(G.mdiag[i * blockSize * blockSize]), Glf1, Glf);
        A.mmAdd(blockSize, &(G.mdiag[i * blockSize * blockSize]), Glf,
                &(G.mdiag[i * blockSize * blockSize]));

        delete[] Glf;
        delete[] Glf1;

        
        // std::cout << "After 1" << i << ". INSIDE second loop (performed left to right)\n";
        // std::cout << "printing A_updiag from CUDA: \n"; 
        // printMatrix(A.updiag, blockSize , blockSize);
        // std::cout << "printing A_mdiag from CUDA: \n"; 
        // printMatrix(A.mdiag, blockSize , blockSize);
        // printMatrix(&(A.mdiag[blockSize*blockSize]), blockSize , blockSize);
        // std::cout << "printing A_lodiag from CUDA: \n"; 
        // printMatrix(A.lodiag, blockSize , blockSize);;
        // std::cout << "printing G_mdiag from CUDA: \n"; 
        // printMatrix(G.mdiag, blockSize , blockSize);
        // printMatrix(&(G.mdiag[blockSize*blockSize]), blockSize , blockSize);
        // std::cout << "----------------------------------------------------------------------- \n"; 
        
    }
}

