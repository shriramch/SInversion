import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd

# input 
# Use IQR method to remove outliers
def remove_outliers(df):
    return df.drop(range(5))

# arguments 
matrix_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
block_sizes = [8, 16, 32, 64, 128]
file_list = ["data_rgf1.r0", ["data_rgf2.r0","data_rgf2.r1"], "data_rgf1_cuda.r0", ["data_rgf2_cuda.r0","data_rgf2_cuda.r1"]]
algo_names = ["Rgf1", "Rgf2", "Rgf1_cuda", "Rgf2_cuda"]
file_path = "output_logs_davinci_old/set1"
markers = ['o','s','^','D', '+']


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

# for a fixed block size, time vs matrix size for different algorithm in one plot
# in total 5 plots
def lineplot_time_block(): 
    
    for block_size in block_sizes:
        plt.figure(figsize=(8, 6))
        plt.grid()
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
                    ci = sns.utils.ci(sns.algorithms.bootstrap(df['time'].values / 1000, func=np.median, n_boot=1000))
                    data_ci_low.append(ci[0])
                    data_ci_high.append(ci[1])
                # rgf1 and rgf1_cuda
                else:
                    data_path = os.path.join(matrix_block, file)
                    df = pd.read_csv(data_path, delim_whitespace=True, comment='#',usecols=['time'])
                    df = remove_outliers(df)
                    median = np.median(df['time'].values)
                    data_y.append(median/1000)
                    ci = sns.utils.ci(sns.algorithms.bootstrap(df['time'].values / 1000, func=np.median, n_boot=1000))
                    data_ci_low.append(ci[0])
                    data_ci_high.append(ci[1])
            sns.lineplot(x=matrix_sizes, y=data_y, label=algo_names[i], marker=markers[i])
            plt.fill_between(matrix_sizes, data_ci_low, data_ci_high, alpha=0.2)
        
        # Add labels and title
        plt.xlabel('Matrix Size')
        plt.ylabel('Time (ms)')
        # plt.title(f"Time for different matrix sizes with block size {block_size}")
        plt.yscale('log')
        plt.xscale('log')
        plt.xticks(matrix_sizes, matrix_sizes)
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.85))
        # Display the plot
        plt.savefig(f"./figures/lineplot_time_block{block_size}.png")
        plt.savefig(f"./figures/lineplot_time_block{block_size}.eps")
        plt.clf()

# for a fixed block size, performance vs matrix size for different algorithm in one plot
# in total 5 plots
def lineplot_performance_block(): 
    
    for block_size in block_sizes:
        plt.figure(figsize=(8, 6))
        plt.grid()
        for i, file in enumerate(file_list): # different algorithm
            data_y = []
            perf_list = []

            for matrix_size in matrix_sizes:
                matrix_block = os.path.join(file_path, str(matrix_size) + "_" + str(block_size))
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
                    # perf_list.append(flops_count / cycle_count)
                    perf_list.append((flops_count / (median/1000000))/1000000000) # micro second to second

                # rgf1 and rgf1_cuda
                else:
                    data_path = os.path.join(matrix_block, file)
                    df = pd.read_csv(data_path, delim_whitespace=True, comment='#',usecols=['time'])
                    df = remove_outliers(df)
                    median = np.median(df['time'].values)
                    data_y.append(median)
                    flops_count, cycle_count = rgf1_flops_cycles(matrix_size, block_size, median)
                    # perf_list.append(flops_count / cycle_count)
                    perf_list.append((flops_count / (median/1000000)) / 1000000000)
            sns.lineplot(x=matrix_sizes, y=perf_list, label=algo_names[i], marker=markers[i])

        # Add labels and title
        plt.xlabel('Matrix Size')
        plt.ylabel('Performance (Gflops/second)')
        # plt.title(f"performance for different matrix sizes with block size {block_size}")
        # plt.yscale('log')
        plt.xscale('log')
        plt.xticks(matrix_sizes, matrix_sizes)
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.85))
        # Display the plot
        plt.savefig(f"./figures/lineplot_performance_block{block_size}.png")
        plt.savefig(f"./figures/lineplot_performance_block{block_size}.eps")
        plt.clf()


# for a fixed algorithm,  performance vs matrix size for different block sizes in one plot
def lineplot_time_algorithm():
    
    for idx, file in enumerate(file_list): # different algorithm
        plt.figure(figsize=(8, 6))
        plt.grid()
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
                ci = sns.utils.ci(sns.algorithms.bootstrap(df['time'].values / 1000, func=np.median, n_boot=1000))
                data_y.append(median)
                data_ci_low.append(ci[0])
                data_ci_high.append(ci[1])

            sns.lineplot(x=matrix_sizes, y=data_y, label="Blocksize="+str(block_size), marker=markers[i])
            plt.fill_between(matrix_sizes, data_ci_low, data_ci_high, alpha=0.2)
        
        plt.xlabel('Matrix Size')
        plt.ylabel('Time (ms)')
        # plt.title(f'{algo_names[idx]}: Time vs. Matrix Size')
        plt.yscale('log')
        plt.xscale('log')
        plt.xticks(matrix_sizes, matrix_sizes)
        plt.legend()
        plt.savefig(f"./figures/lineplot_time_matrix_{algo_names[idx]}.png")
        plt.savefig(f"./figures/lineplot_time_matrix_{algo_names[idx]}.eps")
        plt.clf()

def lineplot_performance_algorithm():
    for idx, file in enumerate(file_list): # different algorithm
        plt.figure(figsize=(8, 6))
        plt.grid()
        for i, block_size in enumerate(block_sizes):
            perf_list = []
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
                        continue
                    max_df = pd.DataFrame({
                        'time': df1['time'].combine(df2['time'], lambda x1, x2: max(x1, x2)),
                    })
                    df = max_df
                    df = remove_outliers(df)
                    median = np.median(df['time'].values) # microsecond to millisecond
                    flops_count, cycle_count = rgf2_flops_cycles(matrix_size, block_size, median)
                    perf_list.append((flops_count / (median/1000000))/1000000000) # micro second to second

                else:
                    data_path = os.path.join(matrix_block, file)
                    df = pd.read_csv(data_path, delim_whitespace=True, comment='#',usecols=['time'])
                    df = remove_outliers(df)
                    median = np.median(df['time'].values) # microsecond to millisecond
                    flops_count, cycle_count = rgf1_flops_cycles(matrix_size, block_size, median)
                    perf_list.append((flops_count / (median/1000000))/1000000000) # micro second to second

            sns.lineplot(x=matrix_sizes, y=perf_list, label="Blocksize="+str(block_size), marker=markers[i])
        
        plt.xlabel('Matrix Size')
        plt.ylabel('Performance (Gflops/second)')
        # plt.title(f'{algo_names[idx]}: Time vs. Matrix Size')
        plt.yscale('log')
        plt.xscale('log')
        plt.xticks(matrix_sizes, matrix_sizes)
        plt.legend()
        plt.savefig(f"./figures/lineplot_performance_matrix_{algo_names[idx]}.png")
        plt.savefig(f"./figures/lineplot_performance_matrix_{algo_names[idx]}.eps")
        plt.clf()


def boxplot_performance(): 
    plt.figure(figsize=(8, 6))
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
                    flops_count, cycle_count = rgf2_flops_cycles(matrix_size, block_size, 0)
                    perf_data = [(flops_count / (t/1000000)/1000000000) for t in df['time'].values]
                    perf_list.append(perf_data)
                # rgf1 and rgf1_cuda
                else:
                    data_path = os.path.join(matrix_block, file)
                    df = pd.read_csv(data_path, delim_whitespace=True, comment='#',usecols=['time'])
                    df = remove_outliers(df)
                    data_y.append(df['time'].values/1000)
                    flops_count, cycle_count = rgf1_flops_cycles(matrix_size, block_size, 0)
                    perf_data = [(flops_count / (t/1000000))/1000000000 for t in df['time'].values]
                    perf_list.append(perf_data)
            sns.boxplot(data=perf_list, showfliers=True)

            plt.xticks(range(len(matrix_sizes)), matrix_sizes)
            # Add labels and title
            plt.xlabel('Matrix size')
            plt.ylabel('Performance (Gflops/second)')
            # plt.title(f"{algo_names[i]}: performance for different matrix sizes with block size {block_size}")
            plt.savefig(f"./figures/boxplot_performance_block{block_size}_{algo_names[i]}.png")
            plt.savefig(f"./figures/boxplot_performance_block{block_size}_{algo_names[i]}.eps")
            plt.clf()


# fix matrix size 32k, performance vs block size;  for different algorithm
def lineplot_32K_performance_blocksize(): 
    plt.figure(figsize=(8, 6))
    plt.grid()
    matrix_size = 32768
    for i, file in enumerate(file_list): # different algorithm
        data_y = []
        perf_list = []
        for block_size in block_sizes:
            matrix_block = os.path.join(file_path, str(matrix_size) + "_" + str(block_size))
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
                # perf_list.append(flops_count / cycle_count)
                perf_list.append((flops_count / (median/1000000))/1000000000) # micro second to second
            # rgf1 and rgf1_cuda
            else:
                data_path = os.path.join(matrix_block, file)
                df = pd.read_csv(data_path, delim_whitespace=True, comment='#',usecols=['time'])
                df = remove_outliers(df)
                median = np.median(df['time'].values)
                data_y.append(median)
                flops_count, cycle_count = rgf1_flops_cycles(matrix_size, block_size, median)
                # perf_list.append(flops_count / cycle_count)
                perf_list.append((flops_count / (median/1000000))/1000000000)
        sns.lineplot(x=block_sizes, y=perf_list, label=algo_names[i], marker=markers[i])

    # Add labels and title
    plt.xlabel('Block Size')
    plt.ylabel('Performance (Gflops/second)')
    # plt.title(f"performance for different block sizes with matrix size {matrix_size}")
    # plt.yscale('log')
    plt.xscale('log')
    plt.xticks(block_sizes, block_sizes)
    plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.85))
    # Display the plot
    plt.savefig(f"./figures/lineplot_performance_matrix{matrix_size}.png")
    plt.savefig(f"./figures/lineplot_performance_matrix{matrix_size}.eps")    
    plt.clf()


# not used     
def boxplot_32K_performance_blocksize_view1():
    data = {
    'block size': ['16']*4 + ['128']*4,
    'box': ['rgf1_16', 'rgf2_16', 'rgf1_cuda_16', 'rgf2_cuda_16', 'rgf1_128', 'rgf2_128', 'rgf1_cuda_128', 'rgf2_cuda_128'],
    'value': []  # Replace this with your actual data
    }
    matrix_size = 32768
    block_sizes = [16, 128]
    for block_size in block_sizes:
        data_y = []
        for  i, file in enumerate(file_list):
            matrix_block = os.path.join(file_path, str(matrix_size) + "_" + str(block_size))
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
                flops_count, cycle_count = rgf2_flops_cycles(matrix_size, block_size, 0)
                perf_data = [flops_count / (t/1000000) for t in df['time'].values]
                # perf_list.append(perf_data)
                data['value'].append(perf_data)
            # rgf1 and rgf1_cuda
            else:
                data_path = os.path.join(matrix_block, file)
                df = pd.read_csv(data_path, delim_whitespace=True, comment='#',usecols=['time'])
                df = remove_outliers(df)
                data_y.append(df['time'].values/1000)
                flops_count, cycle_count = rgf1_flops_cycles(matrix_size, block_size, 0)
                perf_data = [flops_count / (t/1000000) for t in df['time'].values]
                # perf_list.append(perf_data)
                data['value'].append(perf_data)
    # plotting
    df = pd.DataFrame(data)
    df = df.explode('value') 
    df['value'] = df['value'].astype(float)
    # Set up the Seaborn style if needed
    sns.set(style="whitegrid")

    # Create the box plot using Seaborn
    plt.figure(figsize=(12, 8))

    # Set the positions of each box
    box_positions = np.arange(len(df['box'].unique())) * 1.5
    # Create the box plot using Seaborn with dodge=False
    box_plot = sns.boxplot(x='box', y='value', hue='block size', data=df, palette="Set3", dodge=False)

    # Add labels and title
    plt.xlabel('Algorithms')
    plt.ylabel('Performance (flops/second)')
    # plt.title(f'performance variance for 2 block sizes with matrix size {matrix_size}')

    # Show legend at the upper right corner outside the plot
    # plt.legend(title='Box', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(title='Block Size', loc='upper left', bbox_to_anchor=(1, 1))

    plt.savefig(f"figures/boxplot_performance_matrix{matrix_size}.png")
    plt.savefig(f"figures/boxplot_performance_matrix{matrix_size}.eps")

# not used
def boxplot_32K_performance_blocksize_view2():
    data = {
    'block size': ['16']*4 + ['128']*4,
    'box': ['rgf1', 'rgf2', 'rgf1_cuda', 'rgf2_cuda', 'rgf1', 'rgf2', 'rgf1_cuda', 'rgf2_cuda'],
    'value': []  # Replace this with your actual data
    }
    matrix_size = 32768
    block_sizes = [16, 128]
    for block_size in block_sizes:
        data_y = []
        for i, file in enumerate(file_list):
            matrix_block = os.path.join(file_path, str(matrix_size) + "_" + str(block_size))
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
                flops_count, cycle_count = rgf2_flops_cycles(matrix_size, block_size, 0)
                perf_data = [flops_count / (t/1000000) for t in df['time'].values]
                # perf_list.append(perf_data)
                data['value'].append(perf_data)
            # rgf1 and rgf1_cuda
            else:
                data_path = os.path.join(matrix_block, file)
                df = pd.read_csv(data_path, delim_whitespace=True, comment='#',usecols=['time'])
                df = remove_outliers(df)
                data_y.append(df['time'].values/1000)
                flops_count, cycle_count = rgf1_flops_cycles(matrix_size, block_size, 0)
                perf_data = [flops_count / (t/1000000) for t in df['time'].values]
                # perf_list.append(perf_data)
                data['value'].append(perf_data)
    # plotting
    df = pd.DataFrame(data)
    df = df.explode('value') 
    df['value'] = df['value'].astype(float)
    # Set up the Seaborn style if needed
    sns.set(style="whitegrid")

    # Create the box plot using Seaborn
    plt.figure(figsize=(10, 6))
    box_plot = sns.boxplot(x='block size', y='value', hue='box', data=df, palette="Set3")

    # Add labels and title
    plt.xlabel('Block Size')
    plt.ylabel('Performance (flops/second)')
    plt.title(f'performance variance for 2 block sizes with matrix size {matrix_size}')

    # Show legend at the upper right corner outside the plot
    # plt.legend(title='Box', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(title='algorithms', loc='upper left')

    plt.savefig(f"figures/boxplot_performance_matrix{matrix_size}.png")
    plt.savefig(f"figures/boxplot_performance_matrix{matrix_size}.eps")


def boxplot_32K_performance_blocksize_view3():
    matrix_size = 32768
    block_size = 128
    perf_list = []
    data_y = []
    for i, file in enumerate(file_list):
        matrix_block = os.path.join(file_path, str(matrix_size) + "_" + str(block_size))
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
            flops_count, cycle_count = rgf2_flops_cycles(matrix_size, block_size, 0)
            perf_data = [(flops_count / (t/1000000))/1000000000 for t in df['time'].values]
            perf_list.append(perf_data)

        # rgf1 and rgf1_cuda
        else:
            data_path = os.path.join(matrix_block, file)
            df = pd.read_csv(data_path, delim_whitespace=True, comment='#',usecols=['time'])
            df = remove_outliers(df)
            data_y.append(df['time'].values/1000)
            flops_count, cycle_count = rgf1_flops_cycles(matrix_size, block_size, 0)
            perf_data = [(flops_count / (t/1000000))/1000000000 for t in df['time'].values]
            perf_list.append(perf_data)
    
    sns.boxplot(data=perf_list, showfliers=True)
    # sns.violinplot(data=perf_list)
    
    plt.xticks(range(4), ['rgf1', 'rgf2', 'rgf1_cuda', 'rgf2_cuda'])
    plt.yscale('log')
    # Add labels and title
    plt.xlabel('Algorithms')
    plt.ylabel('Performance (Gflops/second)')
    plt.savefig(f"./figures/boxplot_performance_block{block_size}_matrix{matrix_size}.png")
    plt.savefig(f"./figures/boxplot_performance_block{block_size}_matrix{matrix_size}.eps")
    plt.clf()


  

# lineplot_time_block()
lineplot_performance_block()
# lineplot_time_algorithm()
lineplot_performance_algorithm()
lineplot_32K_performance_blocksize()
boxplot_32K_performance_blocksize_view3()