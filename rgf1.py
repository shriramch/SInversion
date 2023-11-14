"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Nicolas Vetsch (nvetsch@iis.ee.ethz.ch)
@date: 2023-05

@reference: https://doi.org/10.1063/1.1432117

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np

from quasi.bsparse._base import bsparse


def rgf(
    A: bsparse,
    sym_mat: bool = False,
    save_off_diag: bool = True,
) -> bsparse:
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

    # 0. Inverse of the first block
    G.blocks[0, 0] = np.linalg.inv(A.blocks[0, 0])

    # 1. Forward substitution (performed left to right)
    for i in range(1, A.blockorder, 1):
        G.blocks[i, i] = np.linalg.inv(
            A.blocks[i, i]
            - A.blocks[i, i - 1] @ G.blocks[i - 1, i - 1] @ A.blocks[i - 1, i]
        )

    # 2. Backward substitution (performed right to left)
    for i in range(A.blockorder - 2, -1, -1):
        g_ii = G.blocks[i, i]
        G_lowerfactor = G.blocks[i + 1, i + 1] @ A.blocks[i + 1, i] @ g_ii

        if save_off_diag:
            G.blocks[i + 1, i] = -G_lowerfactor
            if sym_mat:
                G.blocks[i, i + 1] = G.blocks[i + 1, i].T
            else:
                G.blocks[i, i + 1] = -g_ii @ A.blocks[i, i + 1] @ G.blocks[i + 1, i + 1]

        G.blocks[i, i] = g_ii + g_ii @ A.blocks[i, i + 1] @ G_lowerfactor

    return G
