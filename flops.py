import numpy as np

N_list = [256 * (2 ** i) for i in range(10)]
B_list = [8 * (2 ** i) for i in range(5)]

flops_list = np.zeros((10, 5))
cycles_list = np.zeros((10, 5))
perf_list = np.zeros((10, 5))

for i, N in enumerate(N_list):
  for j, B in enumerate(B_list):
    Mf = 2 * B * B * B - B * B
    Sf = B * B
    If = round(2 * B * B * B / 3)

    M = 2 * (N / B - 1) + 4 * (N / B - 2) + 2
    S = (N / B - 1) + 2 * (N / B - 2) + 1
    I = N / B

    flops[i][j] = M * Mf + S * Sf + I * If

    time = int(input())

    # get time from Yifan's script

    cycles[i][j] = time * 3.2 * (10 ** 3)

    perf[i][j] = flops[i][j] / cycles[i][j]

    print ("Flops: ", flops[i][j])
    print ("Cycles: ", cycles[i][j])
    print ("Performance: ", perf[i][j])