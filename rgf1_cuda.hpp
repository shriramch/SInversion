#pragma once
#include "matrices_utils.hpp"
#include <cblas.h>
#include <lapacke.h>

// Declaration of the CUDA-accelerated version of rgf1sided
void rgf1sided_cuda(Matrix &A, Matrix &G, bool sym_mat = false,
                    bool save_off_diag = true);