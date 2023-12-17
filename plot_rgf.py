import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd

# input 
# Use IQR method to remove outliers
def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
    df_no_outliers_iqr = df[~outliers_iqr]
    return df_no_outliers_iqr

# arguments 
matrix_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
block_sizes = [16, 32, 64, 128]
file_list = ["data_rgf1.r0", ["data_rgf2.r0","data_rgf2.r1"], "data_rgf1_cuda.r0", ["data_rgf2_cuda.r0","data_rgf2_cuda.r1"]]
algo_names = ["rgf1", "rgf2", "rgf1_cuda", "rgf2_cuda"]
file_path = "output_logs_20231216_231547"
markers = ['o','s','^','D']

# flops and cycles of rgf1 and rgf1_cuda
def rgf1_flops_cycles(N, B, time):
    Mf = 2 * B * B * B - B * B
    Sf = B * B
    If = round(2 * B * B * B / 3)
    M = 2 * (N / B - 1) + 4 * (N / B - 2) + 2
    S = (N / B - 1) + 2 * (N / B - 2) + 1
    I = N / B

    flops_count = M * Mf + S * Sf + I * If
    cycle_count = time * 3.2 * (10 ** 3)
    return flops_count, cycle_count

# flops and cycles of rgf2 and rgf2_cuda
def rgf2_flops_cycles(N, B, time):
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

    flops_count = M * Mf + S * Sf + I * If
    cycle_count = time * 3.2 * (10 ** 3)
    return flops_count, cycle_count

# time vs matrix size for different block size and algorithm
# similar code with lineplot_performance
def lineplot_time(): 
    for block_size in block_sizes:
        for i, file in enumerate(file_list): # different algorithm
            data_y = []

            data_ci_low = []
            data_ci_high = []
            for matrix_size in matrix_sizes:
                matrix_block = os.path.join(file_path, str(matrix_size) + "_" + str(block_size))
                B = block_size
                N = matrix_size
                if isinstance(file, list): # 2 ranks rgf2 and rgf2_cuda
                    data_path1 = os.path.join(matrix_block, file[0])
                    data_path2 = os.path.join(matrix_block, file[1])
                    try:
                        df1 = pd.read_csv(data_path1, delim_whitespace=True, comment='#',usecols=['time'])
                        df2 = pd.read_csv(data_path2, delim_whitespace=True, comment='#',usecols=['time'])
                    # deal with 256 matrix size and 128 block size; not used now
                    except Exception as e: 
                        data_y.append(0)
                        continue
                    max_df = pd.DataFrame({
                        'time': df1['time'].combine(df2['time'], lambda x1, x2: max(x1, x2)),
                    })
                    df = max_df
                    df = remove_outliers(df)
                    median = np.median(df['time'].values)
                    # print(f"In {data_path1}, matrix_size={matrix_size} block size={block_size}, median is {median}")
                    data_y.append(median/1000)
                    ci = sns.utils.ci(sns.algorithms.bootstrap(df['time'].values / 1000, n_boot=1000))
                    data_ci_low.append(ci[0])
                    data_ci_high.append(ci[1])
                # rgf1 and rgf1_cuda
                else:
                    data_path = os.path.join(matrix_block, file)
                    df = pd.read_csv(data_path, delim_whitespace=True, comment='#',usecols=['time'])
                    df = remove_outliers(df)
                    median = np.median(df['time'].values)
                    data_y.append(median/1000)
                    ci = sns.utils.ci(sns.algorithms.bootstrap(df['time'].values / 1000, n_boot=1000))
                    data_ci_low.append(ci[0])
                    data_ci_high.append(ci[1])
            sns.lineplot(x=matrix_sizes, y=data_y, label=algo_names[i], marker=markers[i])
            plt.fill_between(matrix_sizes, data_ci_low, data_ci_high, alpha=0.2)
        # Add labels and title
        plt.xlabel('matrix size')
        plt.ylabel('time (ms)')
        plt.title(f"time for different matrix sizes with block size {block_size}")
        plt.yscale('log')
        plt.xscale('log')
        plt.xticks(matrix_sizes, matrix_sizes)
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.85))
        # Display the plot
        plt.savefig(f"./figures/lineplot_time_block{block_size}.png")
        plt.clf()

def lineplot_performance(): 
    for block_size in block_sizes:
        for i, file in enumerate(file_list): # different algorithm
            data_y = []

            flops_list=[]
            cycles_list = []
            perf_list = []

            for matrix_size in matrix_sizes:
                matrix_block = os.path.join(file_path, str(matrix_size) + "_" + str(block_size))
                B = block_size
                N = matrix_size
                if isinstance(file, list): # 2 ranks rgf2 and rgf2_cuda
                    data_path1 = os.path.join(matrix_block, file[0])
                    data_path2 = os.path.join(matrix_block, file[1])
                    try:
                        df1 = pd.read_csv(data_path1, delim_whitespace=True, comment='#',usecols=['time'])
                        df2 = pd.read_csv(data_path2, delim_whitespace=True, comment='#',usecols=['time'])
                    # deal with 256 matrix size and 128 block size; not used now
                    except Exception as e: 
                        data_y.append(0)
                        continue
                    max_df = pd.DataFrame({
                        'time': df1['time'].combine(df2['time'], lambda x1, x2: max(x1, x2)),
                    })
                    df = max_df
                    df = remove_outliers(df)
                    median = np.median(df['time'].values)
                    flops_count, cycle_count = rgf2_flops_cycles(matrix_size, block_size, median)
                    # print(f"In {data_path1}, matrix_size={matrix_size} block size={block_size}, median is {median}")
                    data_y.append(median)
                    flops_list.append(flops_count)
                    cycles_list.append(cycle_count)
                    perf_list.append(flops_count / cycle_count)

                # rgf1 and rgf1_cuda
                else:
                    data_path = os.path.join(matrix_block, file)
                    df = pd.read_csv(data_path, delim_whitespace=True, comment='#',usecols=['time'])
                    df = remove_outliers(df)
                    median = np.median(df['time'].values)
                    data_y.append(median)
                    flops_count, cycle_count = rgf1_flops_cycles(matrix_size, block_size, median)
                    flops_list.append(flops_count)
                    cycles_list.append(cycle_count)
                    perf_list.append(flops_count / cycle_count)
            sns.lineplot(x=matrix_sizes, y=perf_list, label=algo_names[i], marker=markers[i])

        # Add labels and title
        plt.xlabel('matrix size')
        plt.ylabel('performance (flops/cycle)')
        plt.title(f"performance for different matrix sizes with block size {block_size}")
        # plt.yscale('log')
        plt.xscale('log')
        plt.xticks(matrix_sizes, matrix_sizes)
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.85))
        # Display the plot
        plt.savefig(f"./figures/lineplot_performance_block{block_size}.png")
        plt.clf()


# line plot for different algorithms with different matrix size and block size
def lineplot_algorithm():
    for idx, file in enumerate(file_list): # different algorithm
        for i, block_size in enumerate(block_sizes):
            data_y = []
            # confidence interval - to make sure
            data_ci_low = []
            data_ci_high = []
            for matrix_size in matrix_sizes:            
                matrix_block = os.path.join(file_path, str(matrix_size) + "_" + str(block_size))
                if isinstance(file, list): # 2 ranks
                    data_path1 = os.path.join(matrix_block, file[0])
                    data_path2 = os.path.join(matrix_block, file[1])
                    try: # deal with matrix size 256, block size 128
                        df1 = pd.read_csv(data_path1, delim_whitespace=True, comment='#',usecols=['time'])
                        df2 = pd.read_csv(data_path2, delim_whitespace=True, comment='#',usecols=['time'])
                    # deal with 256 matrix size and 128 block size; not used now
                    except Exception as e:
                        data_y.append(0)
                        data_ci_low.append(0)
                        data_ci_high.append(0)
                        continue
                    max_df = pd.DataFrame({
                        'time': df1['time'].combine(df2['time'], lambda x1, x2: max(x1, x2)),
                    })
                    df = max_df
                else:
                    data_path = os.path.join(matrix_block, file)
                    df = pd.read_csv(data_path, delim_whitespace=True, comment='#',usecols=['time'])
                
                df = remove_outliers(df)
                median = np.median(df['time'].values / 1000) # microsecond to millisecond
                ci = sns.utils.ci(sns.algorithms.bootstrap(df['time'].values / 1000, n_boot=1000))
                data_y.append(median)
                data_ci_low.append(ci[0])
                data_ci_high.append(ci[1])

            sns.lineplot(x=matrix_sizes, y=data_y, label="blocksize="+str(block_size), marker=markers[i])
            plt.fill_between(matrix_sizes, data_ci_low, data_ci_high, alpha=0.2)
        
        plt.xlabel('matrix size')
        plt.ylabel('time (ms)')
        plt.title(f'{algo_names[idx]}: Time vs. Matrix Size')
        plt.yscale('log')
        plt.xscale('log')
        plt.xticks(matrix_sizes, matrix_sizes)
        plt.legend()
        plt.savefig(f"./figures/lineplot_time_matrix_{algo_names[idx]}.png")
        plt.clf()



def boxplot_performance(): 
    for block_size in block_sizes:
        for i, file in enumerate(file_list): # different algorithm
            data_y = []

            flops_list=[]
            cycles_list = []
            perf_list = []

            for matrix_size in matrix_sizes:
                matrix_block = os.path.join(file_path, str(matrix_size) + "_" + str(block_size))
                B = block_size
                N = matrix_size
                if isinstance(file, list): # 2 ranks rgf2 and rgf2_cuda
                    data_path1 = os.path.join(matrix_block, file[0])
                    data_path2 = os.path.join(matrix_block, file[1])
                    try:
                        df1 = pd.read_csv(data_path1, delim_whitespace=True, comment='#',usecols=['time'])
                        df2 = pd.read_csv(data_path2, delim_whitespace=True, comment='#',usecols=['time'])
                    # deal with 256 matrix size and 128 block size; not used now
                    except Exception as e: 
                        data_y.append(0)
                        continue
                    max_df = pd.DataFrame({
                        'time': df1['time'].combine(df2['time'], lambda x1, x2: max(x1, x2)),
                    })
                    df = max_df
                    df = remove_outliers(df)
                    data_y.append(df['time'].values/1000)
                    # median = np.median(df['time'].values)
                    # flops_count, cycle_count = rgf2_flops_cycles(matrix_size, block_size, median)
                    # data_y.append(median)
                    # flops_list.append(flops_count)
                    # cycles_list.append(cycle_count)
                    # perf_list.append(flops_count / cycle_count)

                # rgf1 and rgf1_cuda
                else:
                    data_path = os.path.join(matrix_block, file)
                    df = pd.read_csv(data_path, delim_whitespace=True, comment='#',usecols=['time'])
                    df = remove_outliers(df)
                    data_y.append(df['time'].values/1000)
                    # median = np.median(df['time'].values)
                    # data_y.append(median)
                    # flops_count, cycle_count = rgf1_flops_cycles(matrix_size, block_size, median)
                    # flops_list.append(flops_count)
                    # cycles_list.append(cycle_count)
                    # perf_list.append(flops_count / cycle_count)
            # sns.lineplot(x=matrix_sizes, y=perf_list, label=algo_names[i], marker=markers[i])
            sns.boxplot(data=data_y)

            plt.xticks(range(len(matrix_sizes)), matrix_sizes)
            # Add labels and title
            plt.xlabel('matrix size')
            plt.ylabel('time (ms)')
            plt.title(f"time for different matrix sizes with block size {block_size}")
            plt.savefig(f"./figures/boxplot_time_block{block_size}_{algo_names[i]}.png")
            plt.clf()


# lineplot_time()
# lineplot_performance()
boxplot_performance()
# lineplot_algorithm()
