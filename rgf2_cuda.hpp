#pragma once
#include "matrices_utils.hpp"
#include <assert.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>

class Matrix; // Forward declaration

void rgf2sided_cuda(Matrix &A, Matrix &G, bool sym_mat = false,
                    bool save_off_diag = true);
void rgf2sided_upperprocess_cuda(Matrix &input_A, Matrix &input_G, int nblocks_2,
                            bool sym_mat = false, bool save_off_diag = true);
void rgf2sided_lowerprocess_cuda(Matrix &input_A, Matrix &input_G, int nblocks_2,
                            bool sym_mat = false, bool save_off_diag = true);
