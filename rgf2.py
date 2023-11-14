import numpy as np
from mpi4py import MPI

# functions that can do BLAS MMM and BLAS MI
# upper process -> replicate lower process easily
# rgf2sided


def rgf2sided(
    A_bloc_diag: np.ndarray,
    A_bloc_upper: np.ndarray,
    A_bloc_lower: np.ndarray,
    sym_mat: bool = False,
    save_off_diag: bool = True,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """Extension of the RGF algorithm performing block-tridiagonal selected
    inversion using 2 processes meeting in the middle of the matrix traversal.

    Parameters
    ----------
    A_bloc_diag : np.ndarray
        Diagonal blocks of the matrix A.
    A_bloc_upper : np.ndarray
        Upper off-diagonal blocks of the matrix A.
    A_bloc_lower : np.ndarray
        Lower off-diagonal blocks of the matrix A.

    Returns
    -------
    G_diag_blocks : np.ndarray
        Diagonal blocks of the matrix G.
    G_upper_blocks : np.ndarray
        Upper off-diagonal blocks of the matrix G.
    G_lower_blocks : np.ndarray
        Lower off-diagonal blocks of the matrix G.
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    nblocks_2 = int(A_bloc_diag.shape[0] / 2)

    G_diagblk = np.zeros_like(A_bloc_diag)
    G_upperblk = np.zeros_like(A_bloc_upper)
    G_lowerblk = np.zeros_like(A_bloc_lower)

    if comm_rank == 0:
        (
            G_diagblk[0:nblocks_2,],
            G_upperblk[0:nblocks_2,],
            G_lowerblk[0:nblocks_2,],
        ) = rgf2sided_upperprocess(
            A_bloc_diag[0:nblocks_2,],
            A_bloc_upper[0:nblocks_2,],
            A_bloc_lower[0:nblocks_2,],
            sym_mat,
            save_off_diag,
        )

        G_diagblk[nblocks_2:,] = comm.recv(source=1, tag=0)
        G_upperblk[nblocks_2:,] = comm.recv(source=1, tag=1)
        G_lowerblk[nblocks_2:,] = comm.recv(source=1, tag=2)

    elif comm_rank == 1:
        (
            G_diagblk[nblocks_2:,],
            G_upperblk[nblocks_2 - 1 :,],
            G_lowerblk[nblocks_2 - 1 :,],
        ) = rgf2sided_lowerprocess(
            A_bloc_diag[nblocks_2:,],
            A_bloc_upper[nblocks_2 - 1 :,],
            A_bloc_lower[nblocks_2 - 1 :,],
            sym_mat,
            save_off_diag,
        )

        comm.send(G_diagblk[nblocks_2:,], dest=0, tag=0)
        comm.send(G_upperblk[nblocks_2:,], dest=0, tag=1)
        comm.send(G_lowerblk[nblocks_2:,], dest=0, tag=2)

    return G_diagblk, G_upperblk, G_lowerblk


def rgf2sided_upperprocess(
    A_diagblk_leftprocess: np.ndarray,
    A_upperblk_leftprocess: np.ndarray,
    A_lowerblk_leftprocess: np.ndarray,
    sym_mat: bool = False,
    save_off_diag: bool = True,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """Left process of the 2-sided RGF algorithm. Array traversal is done from
    left to right.

    Parameters
    ----------
    A_diagblk_leftprocess : np.ndarray
        Diagonal blocks of the matrix A.
    A_upperblk_leftprocess : np.ndarray
        Upper off-diagonal blocks of the matrix A.
    A_lowerblk_leftprocess : np.ndarray
        Lower off-diagonal blocks of the matrix A.

    Returns
    -------
    G_diagblk_leftprocess : np.ndarray
        Diagonal blocks of the matrix G.
    G_upperblk_leftprocess : np.ndarray
        Upper off-diagonal blocks of the matrix G.
    G_lowerblk_leftprocess : np.ndarray
        Lower off-diagonal blocks of the matrix G.
    """

    comm = MPI.COMM_WORLD

    nblocks = A_diagblk_leftprocess.shape[0]
    blockSize = A_diagblk_leftprocess.shape[1]

    G_diagblk_leftprocess = np.zeros(
        (nblocks + 1, blockSize, blockSize), dtype=A_diagblk_leftprocess.dtype
    )
    G_upperblk_leftprocess = np.zeros_like(A_upperblk_leftprocess)
    G_lowerblk_leftprocess = np.zeros_like(A_lowerblk_leftprocess)

    # Initialisation of g
    G_diagblk_leftprocess[0,] = np.linalg.inv(A_diagblk_leftprocess[0,])

    # Forward substitution
    for i in range(1, nblocks):
        G_diagblk_leftprocess[i,] = np.linalg.inv(
            A_diagblk_leftprocess[i,]
            - A_lowerblk_leftprocess[i - 1,]
            @ G_diagblk_leftprocess[i - 1,]
            @ A_upperblk_leftprocess[i - 1,]
        )

    # Communicate the left connected block and receive the right connected block
    comm.send(G_diagblk_leftprocess[nblocks - 1,], dest=1, tag=0)
    G_diagblk_leftprocess[nblocks,] = comm.recv(source=1, tag=0)

    # Connection from both sides of the full G
    G_diagblk_leftprocess[nblocks - 1,] = np.linalg.inv(
        A_diagblk_leftprocess[nblocks - 1,]
        - A_lowerblk_leftprocess[nblocks - 2,]
        @ G_diagblk_leftprocess[nblocks - 2,]
        @ A_upperblk_leftprocess[nblocks - 2,]
        - A_upperblk_leftprocess[nblocks - 1,]
        @ G_diagblk_leftprocess[nblocks,]
        @ A_lowerblk_leftprocess[nblocks - 1,]
    )

    # Compute the shared off-diagonal upper block
    G_lowerblk_leftprocess[nblocks - 1,] = (
        -G_diagblk_leftprocess[nblocks,]
        @ A_lowerblk_leftprocess[nblocks - 1,]
        @ G_diagblk_leftprocess[nblocks - 1,]
    )
    if sym_mat:
        G_upperblk_leftprocess[nblocks - 1,] = G_lowerblk_leftprocess[nblocks - 1,].T
    else:
        G_upperblk_leftprocess[nblocks - 1,] = (
            -G_diagblk_leftprocess[nblocks - 1,]
            @ A_upperblk_leftprocess[nblocks - 1,]
            @ G_diagblk_leftprocess[nblocks,]
        )

    # Backward substitution
    for i in range(nblocks - 2, -1, -1):
        g_ii = G_diagblk_leftprocess[i,]
        G_lowerfactor = (
            G_diagblk_leftprocess[i + 1,] @ A_lowerblk_leftprocess[i,] @ g_ii
        )

        if save_off_diag:
            G_lowerblk_leftprocess[i,] = -G_lowerfactor
            if sym_mat:
                G_upperblk_leftprocess[i,] = G_lowerblk_leftprocess[i,].T
            else:
                G_upperblk_leftprocess[i,] = (
                    -g_ii @ A_upperblk_leftprocess[i,] @ G_diagblk_leftprocess[i + 1,]
                )

        G_diagblk_leftprocess[i,] = (
            g_ii + g_ii @ A_upperblk_leftprocess[i,] @ G_lowerfactor
        )

    return (
        G_diagblk_leftprocess[:nblocks,],
        G_upperblk_leftprocess,
        G_lowerblk_leftprocess,
    )


def rgf2sided_lowerprocess(
    A_diagblk_rightprocess: np.ndarray,
    A_upperblk_rightprocess: np.ndarray,
    A_lowerbk_rightprocess: np.ndarray,
    sym_mat: bool = False,
    save_off_diag: bool = True,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """Right process of the 2-sided RGF algorithm. Array traversal is done from
    right to left.

    Parameters
    ----------
    A_diagblk_rightprocess : np.ndarray
        Diagonal blocks of the matrix A.
    A_upperblk_rightprocess : np.ndarray
        Upper off-diagonal blocks of the matrix A.
    A_lowerbk_rightprocess : np.ndarray
        Lower off-diagonal blocks of the matrix A.

    Returns
    -------
    G_diagblk_rightprocess : np.ndarray
        Diagonal blocks of the matrix G.
    G_upperblk_rightprocess : np.ndarray
        Upper off-diagonal blocks of the matrix G.
    G_lowerblk_rightprocess : np.ndarray
        Lower off-diagonal blocks of the matrix G.
    """

    comm = MPI.COMM_WORLD

    nblocks = A_diagblk_rightprocess.shape[0]
    blockSize = A_diagblk_rightprocess.shape[1]

    G_diagblk_rightprocess = np.zeros(
        (nblocks + 1, blockSize, blockSize), dtype=A_diagblk_rightprocess.dtype
    )
    G_upperblk_rightprocess = np.zeros_like(A_upperblk_rightprocess)
    G_lowerblk_rightprocess = np.zeros_like(A_lowerbk_rightprocess)

    # Initialisation of g
    G_diagblk_rightprocess[-1,] = np.linalg.inv(A_diagblk_rightprocess[-1,])

    # Forward substitution
    for i in range(nblocks - 1, 0, -1):
        G_diagblk_rightprocess[i,] = np.linalg.inv(
            A_diagblk_rightprocess[i - 1,]
            - A_upperblk_rightprocess[i,]
            @ G_diagblk_rightprocess[i + 1,]
            @ A_lowerbk_rightprocess[i,]
        )

    # Communicate the right connected block and receive the left connected block
    G_diagblk_rightprocess[0,] = comm.recv(source=0, tag=0)
    comm.send(G_diagblk_rightprocess[1,], dest=0, tag=0)

    # Connection from both sides of the full G
    G_diagblk_rightprocess[1,] = np.linalg.inv(
        A_diagblk_rightprocess[0,]
        - A_lowerbk_rightprocess[0,]
        @ G_diagblk_rightprocess[0,]
        @ A_upperblk_rightprocess[0,]
        - A_upperblk_rightprocess[1,]
        @ G_diagblk_rightprocess[2,]
        @ A_lowerbk_rightprocess[1,]
    )

    # Compute the shared off-diagonal upper block
    G_lowerblk_rightprocess[0,] = (
        G_diagblk_rightprocess[1,]
        @ A_lowerbk_rightprocess[0,]
        @ G_diagblk_rightprocess[0,]
    )
    if sym_mat:
        G_upperblk_rightprocess[0,] = G_lowerblk_rightprocess[0,].T
    else:
        G_upperblk_rightprocess[0,] = (
            G_diagblk_rightprocess[0,]
            @ A_upperblk_rightprocess[0,]
            @ G_diagblk_rightprocess[1,]
        )

    # Backward substitution
    for i in range(1, nblocks):
        g_ii = G_diagblk_rightprocess[i + 1,]
        G_lowerfactor = g_ii @ A_lowerbk_rightprocess[i,] @ G_diagblk_rightprocess[i,]

        if save_off_diag:
            G_lowerblk_rightprocess[i,] = -G_lowerfactor
            if sym_mat:
                G_upperblk_rightprocess[i,] = G_lowerblk_rightprocess[i,].T
            else:
                G_upperblk_rightprocess[i,] = (
                    -G_diagblk_rightprocess[i,] @ A_upperblk_rightprocess[i,] @ g_ii
                )

        G_diagblk_rightprocess[i + 1,] = (
            g_ii + G_lowerfactor @ A_upperblk_rightprocess[i,] @ g_ii
        )

    return G_diagblk_rightprocess[1:,], G_upperblk_rightprocess, G_lowerblk_rightprocess
