import rgf1
import numpy as np

def print_float_matrix(matrix):
    np.set_printoptions(precision=8, suppress=True)

    for row in matrix:
        print(" ".join(f"{element:8.8f}" for element in row))

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
    nblocks, bsize = 4, 4

    A, A_diag, A_upper, A_lower = generate_matrices(nblocks, bsize, 89)

    # print_float_matrix(A)
    # print("Diagonal : ")
    # print(A_diag)
    # print("Upper : ")
    # print(A_upper)
    # print("lower : ")
    # print(A_lower)

    np.set_printoptions(precision=8, suppress=True)
    with open('test.txt', 'w') as file:
        file.write(f'n = {nblocks*bsize}, blocksize = {bsize}\n')
        np.savetxt(file, A, fmt='%8.8f')


    # calling RGF algoriithm
    G_diag, G_lower, G_upper = rgf1.rgf(A_diag, A_lower, A_upper, sym_mat=False, save_off_diag=True)

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

    print()
    print("********************************")
    print("G Diagonal : ")
    print(G_diag)
    print("G Upper : ")
    print(G_upper)
    print("G lower : ")
    print(G_lower)
