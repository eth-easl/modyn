import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plotting.common.common import *


def plot_baravg(pipeline_log, ax, trigger, partition_size=None, num_workers=None, storage_retrieval_threads=None):
    data = []
    for filename, pipeline in pipeline_log:
        if "trainer_log" not in pipeline["supervisor"]["triggers"][trigger]:
            print(f"trainer_log missing in {filename}")
            continue

        if storage_retrieval_threads is not None and pipeline["configuration"]["modyn_config"]["storage"]["retrieval_threads"] != storage_retrieval_threads:
            continue

        if partition_size is not None and pipeline["configuration"]["pipeline_config"]["training"]["selection_strategy"]["maximum_keys_in_memory"] != partition_size:
            continue

        relevant_data = pipeline["supervisor"]["triggers"][trigger]["trainer_log"]["epochs"][0]
        meta_data = pipeline["configuration"]["pipeline_config"]["training"]

        if num_workers is not None and meta_data['dataloader_workers'] not in num_workers:
            continue
        total_fb = relevant_data["TotalFetchBatch"] / 1000
        train_minus_fb = pipeline["supervisor"]["triggers"][trigger]["trainer_log"]["total_train"] / 1000 - total_fb

        x = f"{meta_data['dataloader_workers']}/{meta_data['num_prefetched_partitions']}/{meta_data['parallel_prefetch_requests']}"

        data.append([x, total_fb, train_minus_fb])

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
    data_df = pd.DataFrame(data, columns=["x", "Data Fetch Time", "Other Time"])
    data_df.plot(kind='bar', stacked=True, x="x", ax=ax)

    ax.set_xlabel("Workers / Prefetched Partitions / Parallel Requests")
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.set_ylabel("Time (s)")
    ax.get_legend().set_visible(False)

    ax.set_title(f"Data Stalls vs Training Time (Partition Size = {partition_size})")

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
    num_workers = [8,16]
    plot_baravg(data, ax, "0", partition_size=partition_size, num_workers=num_workers)

    HATCH_WIDTH()
    FIG_LEGEND(fig)

    Y_GRID(ax)
    HIDE_BORDERS(ax)

    plot_path = os.path.join(plot_dir, f"train_fetch_{partition_size}")
    SAVE_PLOT(plot_path)
    PRINT_PLOT_PATHS()