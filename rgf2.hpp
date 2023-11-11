#pragma once
#include <iostream>
#include <fstream>
#include <assert.h>
#include <mpi.h>
#include "matrices_utils.hpp"

class Matrix; // Forward declaration

void rgf2sided(Matrix &A, Matrix &G, bool sym_mat = false, bool save_off_diag = true
               );
void rgf2sided_upperprocess(Matrix &A, Matrix &G, int nblocks_2, bool sym_mat = false,
                            bool save_off_diag = true);
void rgf2sided_lowerprocess(Matrix &A, Matrix &G, int nblocks_2, bool sym_mat = false,
                            bool save_off_diag = true);
