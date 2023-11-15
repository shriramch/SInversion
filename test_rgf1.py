import rgf1
import numpy as np


def generate_matrices(nblocks, bsize, seed):
    rng = np.random.default_rng(seed)

    A_diag = np.empty((nblocks, bsize, bsize), dtype=np.float32)
    A_upper = np.empty((nblocks - 1, bsize, bsize), dtype=np.float32)
    A_lower = np.empty((nblocks - 1, bsize, bsize), dtype=np.float32)

    A = rng.random((nblocks * bsize, nblocks * bsize))

    sums = np.sum(np.abs(A), axis=1)
    np.fill_diagonal(A, sums)

    for i in range(nblocks):
        if i > 0:
            A_lower[i - 1] = A[i * bsize : (i + 1) * bsize, (i - 1) * bsize : i * bsize]
        A_diag[i] = A[i * bsize : (i + 1) * bsize, i * bsize : (i + 1) * bsize]
        if i < nblocks - 1:
            A_upper[i] = A[
                i * bsize : (i + 1) * bsize, (i + 1) * bsize : (i + 2) * bsize
            ]

        for j in range(0, i - 1):
            A[i * bsize : (i + 1) * bsize, j * bsize : (j + 1) * bsize] = 0
        for j in range(i + 2, nblocks):
            A[i * bsize : (i + 1) * bsize, j * bsize : (j + 1) * bsize] = 0

    return A, A_diag, A_upper, A_lower


if __name__ == "__main__":
    nblocks, bsize = 5, 1

    # A, A_diag, A_upper, A_lower = generate_matrices(nblocks, bsize, 42)
    A = np.asarray(
        [
            [2.86297779, 0.97562235, 0.0, 0.0, 0.0],
            [0.97562235, 3.10132593, 0.92676499, 0.0, 0.0],
            [0.0, 0.92676499, 3.20760395, 0.06381726, 0.0],
            [0.0, 0.0, 0.06381726, 2.30493634, 0.89312112],
            [0.0, 0.0, 0.0, 0.89312112, 3.75481635],
        ]
    )
    A_diag = np.asarray(
        [[[2.8629777]], [[3.101326]], [[3.207604]], [[2.3049364]], [[3.7548163]]]
    )
    A_upper = np.asarray(
        [[[0.97562236]], [[0.92676497]], [[0.06381726]], [[0.8931211]]]
    )
    A_lower = np.asarray(
        [[[0.97562236]], [[0.92676497]], [[0.06381726]], [[0.8931211]]]
    )
    # print(A, A_diag, A_upper, A_lower)
    # exit()
    G_diag, G_lower, G_upper = rgf1.rgf(
        A_diag, A_lower, A_upper, sym_mat=True, save_off_diag=True
    )

    np.set_printoptions(formatter={"all": lambda x: str(x)})

    with open("test.txt", "w") as f:
        f.write(str(A).replace("[", "").replace("]", ""))

    # Validate
    inv_A = np.linalg.inv(A)
    for i in range(nblocks):
        if i > 0:
            assert np.allclose(
                inv_A[i * bsize : (i + 1) * bsize, (i - 1) * bsize : i * bsize],
                G_lower[i - 1],
            )
        assert np.allclose(
            inv_A[i * bsize : (i + 1) * bsize, i * bsize : (i + 1) * bsize], G_diag[i]
        )
        if i < nblocks - 1:
            assert np.allclose(
                inv_A[i * bsize : (i + 1) * bsize, (i + 1) * bsize : (i + 2) * bsize],
                G_upper[i],
            )

    with open("testo.txt", "w") as f:
        f.write(str(G_diag))
        f.write("\n")
        f.write(str(G_upper))
        f.write("\n")
        f.write(str(G_lower))
