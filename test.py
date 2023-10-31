"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from os import environ
environ['OMP_NUM_THREADS'] = '1'

# from sinv.algorithms import rgf2s
# from sinv import utils

import numpy as np
import pytest
# from mpi4py import MPI

SEED = 63

def generateRandomNumpyMat(
    matrice_size: int, 
    is_complex: bool = False,
    is_symmetric: bool = False,
    seed: int = None
) -> np.ndarray:
    """ Generate a dense matrix of shape: (matrice_size x matrice_size) filled 
    with random numbers. The matrice may be complex or real valued.

    Parameters
    ----------
    matrice_size : int
        Size of the matrice to generate.
    is_complex : bool, optional
        Whether the matrice should be complex or real valued. The default is False.
    is_symmetric : bool, optional
        Whether the matrice should be symmetric or not. The default is False.
    seed : int, optional
        Seed for the random number generator. The default is no seed.
        
    Returns
    -------
    A : np.ndarray
        The generated matrice.
    """
    if seed is not None:
        np.random.seed(seed)
        
    A = np.zeros((matrice_size, matrice_size))

    if is_complex:
        A = np.random.rand(matrice_size, matrice_size)\
               + 1j * np.random.rand(matrice_size, matrice_size)
    else:
        A = np.random.rand(matrice_size, matrice_size)
        
    if is_symmetric:
        A = A + A.T
        
    return A



def generateBandedDiagonalMatrix(
    matrice_size: int,
    matrice_bandwidth: int, 
    is_complex: bool = False, 
    is_symmetric: bool = False,
    seed: int = None
) -> np.ndarray:
    """ Generate a banded diagonal matrix of shape: matrice_size^2 with a 
    bandwidth = matrice_bandwidth, filled with random numbers.

    Parameters
    ----------
    matrice_size : int
        Size of the matrice to generate.
    matrice_bandwidth : int
        Bandwidth of the matrice to generate.
    is_complex : bool, optional
        Whether the matrice should be complex or real valued. The default is False.
    is_symmetric : bool, optional
        Whether the matrice should be symmetric or not. The default is False.
    seed : int, optional
        Seed for the random number generator. The default is no seed.
        
    Returns
    -------
    A : np.ndarray
        The generated matrice.
    """

    A = generateRandomNumpyMat(matrice_size, is_complex, is_symmetric, seed)
    
    for i in range(matrice_size):
        for j in range(matrice_size):
            if i - j > matrice_bandwidth or j - i > matrice_bandwidth:
                A[i, j] = 0

    return A



def transformToSymmetric(
    A: np.ndarray
) -> np.ndarray:
    """ Make a matrix symmetric by adding its transpose to itself.
    
    Parameters
    ----------
    A : np.ndarray
        The matrix to transform.
        
    Returns
    -------
    A : np.ndarray
        The transformed to symmetric matrix.
    """
    
    return A + A.T



def convertDenseToBlkTridiag(
    A: np.ndarray, 
    blocksize: int
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Converte a square numpy dense matrix to 3 numpy arrays containing the diagonal,
    upper diagonal and lower diagonal blocks.
    
    Parameters
    ----------
    A : np.ndarray
        The matrix to convert.
    blocksize : int
        The size of the blocks.
        
    Returns
    -------
    A_bloc_diag : np.ndarray
        The diagonal blocks.
    A_bloc_upper : np.ndarray
        The upper diagonal blocks.
    A_bloc_lower : np.ndarray
        The lower diagonal blocks.
    """
    
    nblocks = int(np.ceil(A.shape[0]/blocksize))

    A_bloc_diag  = np.zeros((nblocks, blocksize, blocksize), dtype=A.dtype)
    A_bloc_upper = np.zeros((nblocks-1, blocksize, blocksize), dtype=A.dtype)
    A_bloc_lower = np.zeros((nblocks-1, blocksize, blocksize), dtype=A.dtype)

    for i in range(nblocks):
        A_bloc_diag[i, ] = A[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize]
        if i < nblocks-1:
            A_bloc_upper[i, ] = A[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize]
            A_bloc_lower[i, ] = A[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize]

    return A_bloc_diag, A_bloc_upper, A_bloc_lower



def convertBlkTridiagToDense(
    A_diagblk: np.ndarray, 
    A_upperblk: np.ndarray, 
    A_lowerblk: np.ndarray
) -> np.ndarray:
    """ Convert 3 numpy arrays containing the diagonal, upper diagonal and lower
    diagonal blocks to a square numpy dense matrix.
    
    Parameters
    ----------
    A_diagblk : np.ndarray
        The diagonal blocks.
    A_upperblk : np.ndarray
        The upper diagonal blocks.
    A_lowerblk : np.ndarray
        The lower diagonal blocks.
        
    Returns
    -------
    A : np.ndarray
        Dense matrix representation of the input block tridiagonal matrix.
    """
    
    blocksize = A_diagblk.shape[1]
    nblocks   = A_diagblk.shape[0]
    
    A = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A_diagblk.dtype)
    
    for i in range(nblocks):
        A[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = A_diagblk[i, ]
        if i < nblocks-1:
            A[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize] = A_upperblk[i, ]
            A[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize] = A_lowerblk[i, ]
            
    return A


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
    # comm = MPI.COMM_WORLD
    # comm_rank = comm.Get_rank()
    
    bandwidth    = np.ceil(blocksize/2)
    
    A = generateBandedDiagonalMatrix(matrix_size, 
                                                bandwidth, 
                                                is_complex, 
                                                is_symmetric, SEED)
    
    A_diagblk, A_upperblk, A_lowerblk\
        = convertDenseToBlkTridiag(A, blocksize)

    return A, A_diagblk, A_upperblk, A_lowerblk
    
    # G_diagblk, G_upperblk, G_lowerblk\
    #     = rgf2s.rgf2sided(A_diagblk, A_upperblk, A_lowerblk, is_symmetric)

    # A_refsol = np.linalg.inv(A)
    # A_refsol_diagblk, A_refsol_upperblk, A_refsol_lowerblk\
    #     = utils.matu.convertDenseToBlkTridiag(A_refsol, blocksize)
    
    # if comm_rank == 0:
    #     assert np.allclose(A_refsol_diagblk, G_diagblk)\
    #         and np.allclose(A_refsol_upperblk, G_upperblk)\
    #         and np.allclose(A_refsol_lowerblk, G_lowerblk)
        
A, B, C, D = test_rgf2sided(0, 1, 8, 2)

print (A)

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        print (round(A[i][j], 2), end='\t')

    print()

print (D.shape)

for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        for k in range(D.shape[2]):
            print (round(D[i][j][k], 2), end='\t')

        print()
    print()

