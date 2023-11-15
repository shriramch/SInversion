"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Nicolas Vetsch (nvetsch@iis.ee.ethz.ch)
@date: 2023-05

@reference: https://doi.org/10.1063/1.1432117

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np

# from quasi.bsparse._base import bsparse


def rgf(
    A: np.ndarray,
    Al: np.ndarray,
    Au: np.ndarray,
    sym_mat: bool = False,
    save_off_diag: bool = True,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """RGF algorithm performing block-tridiagonal selected inversion on the
    input matrix. Act from upper left to lower right.

    Parameters
    ----------
    A : bsparse
        Input matrix.
    sym_mat : bool, optional
        If True, the input matrix is assumed to be symmetric.
    save_off_diag : bool, optional
        If True, the off-diagonal blocks are saved in the output matrix.

    Returns
    -------
    G : bsparse
        Inverse of the input matrix.
    """

    # Storage for the full backward substitution
    G = A.copy() * np.nan
    Gl = Al.copy() * np.nan
    Gu = Au.copy() * np.nan

    # 0. Inverse of the first block
    G[0] = np.linalg.inv(A[0])

    # print(G[0])

    # 1. Forward substitution (performed left to right)
    for i in range(1, A.shape[0], 1):
        G[i] = np.linalg.inv(A[i] - Al[i - 1] @ G[i - 1] @ Au[i - 1])

    # print(G)

    # 2. Backward substitution (performed right to left)
    for i in range(A.shape[0] - 2, -1, -1):
        g_ii = G[i]
        G_lowerfactor = G[i + 1] @ Al[i] @ g_ii

        # print(G_lowerfactor)

        if save_off_diag:
            Gl[i] = -G_lowerfactor
            if sym_mat:
                Gu[i] = Gl[i].T
            else:
                Gu[i] = -g_ii @ Au[i] @ G[i + 1]

                print(Gu[i])

        G[i] = g_ii + g_ii @ Au[i] @ G_lowerfactor

    return G, Gl, Gu
