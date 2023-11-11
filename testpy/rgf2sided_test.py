"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from os import environ
environ['OMP_NUM_THREADS'] = '1'

# from ...src.sinv.algorithms import rgf2sided
# from ...src.sinv import utils
from sinv.algorithms import rgf2s
from sinv import utils

import numpy as np
import pytest
from mpi4py import MPI

SEED = 63



""" Uniform blocksize tests cases
- Complex and real matrices
- Symmetric and non-symmetric matrices
================================================
| Test n  | Matrice size | Blocksize | nblocks | 
================================================
| Test 1  |     4x4      |     1     |    4    |
| Test 2  |     8x8      |     2     |    4    |
| Test 3  |    12x12     |     3     |    4    |
================================================
| Test 4  |   128x128    |     8     |   16    |
| Test 5  |   128x128    |     16    |    8    |
| Test 6  |   128x128    |     32    |    4    |
================================================ """
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("is_complex", [False, True])
@pytest.mark.parametrize("is_symmetric", [False, True])
@pytest.mark.parametrize(
    "matrix_size, blocksize",
    [
        (4, 1),
        (8, 2),
        (12, 3),
        (128, 8),
        (128, 16),
        (128, 32),
    ]
)
def test_rgf2sided(
    is_complex: bool,
    is_symmetric: bool,
    matrix_size: int,
    blocksize: int
):
    """ Test the RGF 2-Sided algorithm. """
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateFixedMatrixOfSize8()
    # A = utils.matu.generateBandedDiagonalMatrix(matrix_size, 
    #                                             bandwidth, 
    #                                             is_complex, 
    #                                             is_symmetric, SEED)
    
    A_diagblk, A_upperblk, A_lowerblk\
        = utils.matu.convertDenseToBlkTridiag(A, blocksize)
    G_diagblk, G_upperblk, G_lowerblk\
        = rgf2s.rgf2sided(A_diagblk, A_upperblk, A_lowerblk, is_symmetric)

    A_refsol = np.linalg.inv(A)
    A_refsol_diagblk, A_refsol_upperblk, A_refsol_lowerblk\
        = utils.matu.convertDenseToBlkTridiag(A_refsol, blocksize)
    
    if comm_rank == 0:
        print(utils.matu.convertBlkTridiagToDense(G_diagblk, G_upperblk, G_lowerblk))
        assert np.allclose(A_refsol_diagblk, G_diagblk)\
            and np.allclose(A_refsol_upperblk, G_upperblk)\
            and np.allclose(A_refsol_lowerblk, G_lowerblk)
        print("Test passed.")

test_rgf2sided(True, True, 8, 2)