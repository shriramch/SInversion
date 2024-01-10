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

# arguments 
matrix_sizes = 32768
block_sizes = [32, 128, 512, 2048, 8192]
file_list = ["data_rgf1_cuda.r0", "data_rgf2_cuda.r0"]
algo_names = ["Rgf1_cuda", "Rgf2_cuda"]
device_names = ["Davinci", "Ault25", "Ault23"]
log_paths = ["output_logs_davinci_old","output_logs_a100_cuda", "output_logs_v100_cuda"]
# log_paths = ["output_logs_davinci_new"]
markers = ['o','s','^','D', '+']

colors = ['green', 'red', 'purple']
line_type = ['solid', 'dashed']
def lineplot_32K_device_scaling_time_blocksize():
    matrix_size = 32768
    plt.figure(figsize=(8, 6))
    plt.grid()
    for log_idx, log_path in enumerate(log_paths):
        log_path_set2 = os.path.join(log_path, "set2")
        for i, file in enumerate(file_list):
            data_y = []
            perf_list = []
            for block_size in block_sizes:
                matrix_block = os.path.join(log_path_set2, str(matrix_size) + "_" + str(block_size))
                data_path = os.path.join(matrix_block, file)
                try:
                    df = pd.read_csv(data_path, delim_whitespace=True, comment='#',usecols=['time'])
                    df = remove_outliers(df)
                    median = np.median(df['time'].values)
                    data_y.append(median / 1000) # micro second to milli second
                except:
                    # data_y.append(0)
                    data_y.append(np.nan)
            # draw for each file
            sns.lineplot(x=block_sizes, y=data_y, label=f"{algo_names[i]} ({device_names[log_idx]})", marker=markers[i], linestyle=line_type[i], color=colors[log_idx])

    # Add labels and title
    plt.xlabel('Block Size')
    plt.ylabel('Time (ms)')
    # plt.title(f"cuda time for different block sizes with matrix size {matrix_size} on different devices")
    # plt.yscale('log')
    plt.xscale('log', base=2)
    plt.xticks(block_sizes, block_sizes)
    plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.85))
    # Display the plot
    # print("savefig")
    plt.savefig(f"./figures/lineplot_device_scaling_time_matrix{matrix_size}.png")
    plt.savefig(f"./figures/lineplot_device_scaling_time_matrix{matrix_size}.eps")

    plt.clf()      

colors = ['green', 'red', 'purple']
line_type = ['solid', 'dashed']
# fix matrix size 32k, performance vs block size;  for different algorithm
def lineplot_32K_device_scaling_performance_blocksize(): 
    matrix_size = 32768
    plt.figure(figsize=(8, 6))
    plt.grid() 
    for log_idx, log_path in enumerate(log_paths):
        log_path_set2 = os.path.join(log_path, "set2")
        for i, file in enumerate(file_list):
            data_y = []
            perf_list = []
            for block_size in block_sizes:
                matrix_block = os.path.join(log_path_set2, str(matrix_size) + "_" + str(block_size))
                data_path = os.path.join(matrix_block, file)
                try:
                    df = pd.read_csv(data_path, delim_whitespace=True, comment='#',usecols=['time'])
                    df = remove_outliers(df)
                    median = np.median(df['time'].values)
                    data_y.append(median)
                    if file == "data_rgf1_cuda.r0":
                        flops_count, cycle_count = rgf1_flops_cycles(matrix_size, block_size, median)
                    else:
                        flops_count, cycle_count = rgf2_flops_cycles(matrix_size, block_size, median)
                    perf_list.append((flops_count / (median/1000000)) / 1000000000000) # micro second to second
                except:
                    perf_list.append(np.nan)
            # draw for each file
            sns.lineplot(x=block_sizes, y=perf_list, label=f"{algo_names[i]} ({device_names[log_idx]})", marker=markers[i],linestyle=line_type[i], color=colors[log_idx])
    
    # Add labels and title
    plt.xlabel('Block Size')
    plt.ylabel('Performance (Tflops/second)')
    # plt.title(f"cuda performance for different block sizes with matrix size {matrix_size} on different devices")
    # plt.yscale('log')
    plt.xscale('log', base=2)
    plt.xticks(block_sizes, block_sizes)
    plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.85))
    # Display the plot
    plt.savefig(f"./figures/lineplot_device_scaling_performance_matrix{matrix_size}.png")
    plt.savefig(f"./figures/lineplot_device_scaling_performance_matrix{matrix_size}.eps")
    plt.clf()      




file_list = ["data_rgf1_cuda.r0", "data_rgf2_cuda.r0"]
def boxplot_32K_device_scaling_performance_blocksize_view2():
    matrix_size = 32768
    block_size = 2048
    data = {
        'Devices': ['Davinci']*2 + ['A100']*2 + ['V100']*2,
        'box': ['rgf1_cuda (Davinci)', 'rgf2_cuda (Davinci)', 'rgf1_cuda (A100)', 'rgf2_cuda (A100)', 'rgf1_cuda (V100)', 'rgf2_cuda (V100)' ],
        'value': []  # Replace this with your actual data
    }       
    for log_idx, log_path in enumerate(log_paths):
        log_path_set2 = os.path.join(log_path, "set2")
        data_y = []
        perf_data = []
        for i, file in enumerate(file_list):
            matrix_block = os.path.join(log_path_set2, str(matrix_size) + "_" + str(block_size))
            data_path = os.path.join(matrix_block, file)
            # print(data_path)
            df = pd.read_csv(data_path, delim_whitespace=True, comment='#',usecols=['time'])
            df = remove_outliers(df)
            data_y.append(df['time'].values/1000)
            flops_count, cycle_count = rgf1_flops_cycles(matrix_size, block_size, 0)
            perf_data = [(flops_count / (t/1000000))/1000000000000 for t in df['time'].values]
            data['value'].append(perf_data)
        
    df = pd.DataFrame(data)
    df = df.explode('value') 
    df['value'] = df['value'].astype(float)
    # Set up the Seaborn style if needed
    sns.set(style="whitegrid")
    # Create the box plot using Seaborn
    plt.figure(figsize=(10, 6))
    box_plot = sns.boxplot(x='Devices', y='value', hue='box', data=df, palette="Set3")

    # Add labels and title
    plt.xlabel('Devices')
    plt.ylabel('Performance (Tflops/second)')
    # plt.title(f'performance variance')

    plt.legend(title='algorithms', loc='upper left')
    plt.savefig(f"figures/boxplot_device_scaling_performance_matrix{matrix_size}_block{block_size}.png")
    plt.savefig(f"figures/boxplot_device_scaling_performance_matrix{matrix_size}_block{block_size}.eps")



lineplot_32K_device_scaling_time_blocksize()
lineplot_32K_device_scaling_performance_blocksize()
boxplot_32K_device_scaling_performance_blocksize_view2()