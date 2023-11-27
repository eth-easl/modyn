import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plotting.common.common import *


def plot_baravg(pipeline_log, ax, trigger, partition_size=None):
    data = []

    bar_labels = dict()

    for filename, pipeline in pipeline_log:
        if "trainer_log" not in pipeline["supervisor"]["triggers"][trigger]:
            print(f"trainer_log missing in {filename}")
            continue

        if partition_size is not None and pipeline["configuration"]["pipeline_config"]["training"]["selection_strategy"]["maximum_keys_in_memory"] != partition_size:
            continue

        relevant_data = pipeline["supervisor"]["triggers"][trigger]["trainer_log"]["epochs"][0]
        meta_data = pipeline["configuration"]["pipeline_config"]["training"]

        max_fb = relevant_data["MaxFetchBatch"] / 1000
        avg_fb = relevant_data["AvgFetchBatch"] / 1000

        total_fb = relevant_data["TotalFetchBatch"] / 1000
        total_train = pipeline["supervisor"]["triggers"][trigger]["trainer_log"]["total_train"] / 1000
        
        x = f"{meta_data['dataloader_workers']}/{meta_data['num_prefetched_partitions']}/{meta_data['parallel_prefetch_requests']}"

        percentage = round((total_fb / total_train) * 100,1)
        bar_labels[x] = f"{int(total_fb)} ({percentage}%)\n"

        data.append([x, avg_fb, max_fb])

    import functools
    def compare(item1, item2):
        splitted1 = item1[0].split("/")
        workers1 = int(splitted1[0])
        npp1 = int(splitted1[1])
        ppr1 = int(splitted1[2])
        splitted2 = item2[0].split("/")
        workers2 = int(splitted2[0])
        npp2 = int(splitted2[1])
        ppr2 = int(splitted2[2])

        if workers1 < workers2:
            return -1
        if workers1 > workers2:
            return 1
        if npp1 < npp2:
            return -1
        if npp1 > npp2:
            return 1
        if ppr1 < ppr2:
            return -1
        if ppr1 > ppr2:
            return 1
        return 0

    data.sort(key=functools.cmp_to_key(compare))
    data_df = pd.DataFrame(data, columns=["x", "Avg", "Max"])
    test_data_melted = data_df.melt(id_vars="x", value_name = "time", var_name="measure")

    mask = test_data_melted.measure.isin(['Max'])
    scale = test_data_melted[~mask].time.mean()/ test_data_melted[mask].time.mean()
    test_data_melted.loc[mask, 'time'] = test_data_melted.loc[mask, 'time']*scale

    sns.barplot(data=test_data_melted, x="x", y="time", hue="measure", ax=ax)
    bar_label_list = [bar_labels[x._text] for x in ax.get_xticklabels()]
    ax.bar_label(ax.containers[0], labels=bar_label_list, size=11)

    ax.set_xlabel("Workers / Prefetched Partitions / Parallel Requests")
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.set_ylabel("Avg")
    ax2 = ax.twinx()

    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticklabels(np.round(ax.get_yticks()/scale,1))
    ax2.set_ylabel('Max')
    ax.get_legend().set_visible(False)

    #ax.set_xticks(list(x))
    #ax.set_xticklabels([f"{idx + 1}" for idx, _ in enumerate(x)])
    #ax.set_xlabel("Waiting time for next batch (seconds)")

    #ax.set_ylabel("Count")

    ax.set_title("Average and Max Time per Batch")

def load_all_pipelines(data_path):
    all_data = []

    for filename in glob.iglob(data_path + '/**/*.log', recursive=True):
        data = LOAD_DATA(filename)
        all_data.append((filename, data))

    return all_data

if __name__ == '__main__':
    # Idee: Selber plot mit TotalTrain und anteil fetch batch an total train

    data_path, plot_dir = INIT(sys.argv)
    data = load_all_pipelines(data_path)
    fig, ax = plt.subplots(1,1, figsize=(DOUBLE_FIG_WIDTH * 2, DOUBLE_FIG_HEIGHT))
    partition_size = 5000000
    plot_baravg(data, ax, "0", partition_size=partition_size)

    HATCH_WIDTH()
    FIG_LEGEND(fig)

    Y_GRID(ax)
    HIDE_BORDERS(ax)

    plot_path = os.path.join(plot_dir, f"avg_max_{partition_size}")
    SAVE_PLOT(plot_path)
    PRINT_PLOT_PATHS()