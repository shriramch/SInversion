#For now recycled from Advanced System Lab
import os
import sys
import statistics
import seaborn as sns
import pandas as pd

OUTPUT_DEFAULT = "run.txt"

if len(sys.argv) > 1: # you can specify which output file to READ
    target_read_file = sys.argv[1]
    print("Using specified file ", target_read_file)
else:
    target_read_file = OUTPUT_DEFAULT
    print("Using default file ", target_read_file)

data = []

# read file
with open(target_read_file, "r") as f:
    for line in f:
        data.append(line.strip())

if len(data) == 0:
    print("No data found in file ", target_read_file)
    sys.exit(1)

if not data[0].startswith("runs "):
    print("File ", target_read_file, " does not contain runs information")
    sys.exit(1)

run_count = int(data[0][5:])

# parse the data
data_parsed = {}

for i in range(1, len(data), run_count+1):
    val = data[i].split(" ")
    func_name = val[0]
    n_val = val[1]
    func_type = val[2]

    data_parsed.setdefault(func_type, {}).setdefault(func_name, {})[n_val] = []

    for j in range(0, run_count):
        run_data = data[i+j+1].split(" ")
        data_parsed[func_type][func_name][n_val].append((float(run_data[0]), float(run_data[1])))


# compute median and ci
data_aggregated = {}

for func_type in data_parsed.keys():
    for run_func in data_parsed[func_type].keys():
        for run_n in data_parsed[func_type][run_func].keys():
            time_list = []
            size_list = []
            throughput_list = []

            for run_tuple in data_parsed[func_type][run_func][run_n]:
                time_list.append(run_tuple[0])
                size_list.append(run_tuple[1])
                throughput_list.append(run_tuple[1] / run_tuple[0])

            median = statistics.median(throughput_list)
            ci = 1.96 * statistics.stdev(throughput_list) / len(throughput_list) ** 0.5

            time_median = statistics.median(time_list)
            time_ci = 1.96 * statistics.stdev(time_list) / len(time_list) ** 0.5
            size_median = statistics.median(size_list)

            data_aggregated.setdefault(func_type, []).append((run_n, run_func, median, ci, time_median, time_ci, size_median))

# setup plot generation
import matplotlib.pyplot as plt
import matplotlib

#matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family" : "serif",
        "font.serif" : ["Computer Modern Serif"],
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 12,
        "text.usetex": True,
        "axes.unicode_minus": False,
    }
)

paper_rc = {'lines.linewidth': 1.2}                  
sns.set_context(rc = paper_rc)
sns.set_style(style="whitegrid")

# pretty print data
for run_func in data_aggregated.keys():
    df = pd.DataFrame(data_aggregated[run_func], columns=["N", "Impl", "Throughput", "CI", "Time", "Time CI", "Size"])
    impl_list = df["Impl"].unique()

    for impl in impl_list:
        df_hue = df[df["Impl"] == impl]
        print(df_hue)
        print()

    f = sns.lineplot(data=df, x="N", y="Throughput", hue="Impl", legend=False)
    sns.scatterplot(data=df, x="N", y="Throughput", hue="Impl")
    for impl in impl_list:
        df_hue = df[df["Impl"] == impl]
        plt.errorbar(df_hue["N"], df_hue["Throughput"], yerr=df_hue["CI"], fmt='none', color='black')

    f.axis('on')
    f.grid(True)
    f.set(xlabel='Problem size [n]')
    f.set(ylabel='Throughput [Mbytes/s]')
    f.set_title(run_func)
    f.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plot_out/"+run_func+'.pdf', backend='pgf', bbox_inches='tight',pad_inches=0.05)
    plt.clf()