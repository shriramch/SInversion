// #pragma once
#include <cblas.h>
#include <lapacke.h>
#include "matrices_utils.hpp"

// Declaration of the CUDA-accelerated version of rgf1sided
void rgf1sided_cuda(Matrix &A, Matrix &G, bool sym_mat = false, bool save_off_diag = true);