import numpy as np

# N_list = [256 * (2 ** i) for i in range(10)]
# B_list = [8 * (2 ** i) for i in range(5)]

N_list = [131072]
B_list = [64]

flops_list = np.zeros((10, 5))
cycles_list = np.zeros((10, 5))
perf_list = np.zeros((10, 5))

for i, N in enumerate(N_list):
  for j, B in enumerate(B_list):
    Mf = 2 * B * B * B - B * B
    Sf = B * B
    If = round(2 * B * B * B / 3)

    nb = N / B
    nb2 = nb // 2

    M = 2 * nb2 + 8 + 6 * (nb2 - 1)
    S = nb2 + 4 + 3 * (nb2 - 1)
    I = 2 + nb2

    lnb2 = nb - nb2

    M += 2 * (lnb2 - 1) + 8 + 6 * (lnb2 - 1)
    S += (lnb2 - 1) + 2 + 3 * (lnb2 - 1)
    I += 2 + (lnb2 - 1)

    flops_list[i][j] = M * Mf + S * Sf + I * If

    time = int(input())

    # get time from Yifan's script

    cycles_list[i][j] = time * 3.2 * (10 ** 3)

    perf_list[i][j] = flops_list[i][j] / cycles_list[i][j]

    print ("Flops: ", flops_list[i][j])
    print ("Cycles: ", cycles_list[i][j])
    print ("Performance: ", perf_list[i][j])