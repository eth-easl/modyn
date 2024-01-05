import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from plotting.common.common import *
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

def plot_baravg(pipeline_log, ax, trigger, partition_size=None, num_workers=None, storage_retrieval_threads=None):
    data = []
    for filename, pipeline in pipeline_log:

        if trigger not in pipeline["supervisor"]["triggers"]:
            print(f"trigger {trigger} missing in {filename}")
            continue

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


    data.sort(key=functools.cmp_to_key(compare))
    data_df = pd.DataFrame(data, columns=["x", "Data Fetch Time", "Other Time"])
    data_df.plot(kind='bar', stacked=True, x="x", ax=ax, ylim=[0, 375])

    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.set_xlabel("")
    ax.get_legend().set_visible(False)
    ax.bar_label(ax.containers[-1], fmt='%.0f', label_type='edge')

    ax.set_title(f"")

def load_all_pipelines(data_path, dataset=None):
    all_data = []

    for filename in glob.iglob(data_path + '/**/*.log', recursive=True):
        data = LOAD_DATA(filename)

        if dataset is not None and data["configuration"]["pipeline_config"]["data"]["dataset_id"] != dataset:
            continue

        if "local" in data["configuration"]["pipeline_config"]["pipeline"]["name"]:
            continue

        all_data.append((filename, data))

    return all_data

if __name__ == '__main__':
    data_path, plot_dir = INIT(sys.argv)
    all_data = load_all_pipelines(data_path, "criteo_tiny")

    fig, axes = plt.subplots(3, 2, figsize=(DOUBLE_FIG_WIDTH, 1.5 * DOUBLE_FIG_HEIGHT), sharex=True)

    row_vals = [1,2,8] # Threads @ Storage
    column_vals = [100000, 2500000] # Partition Size
    headings = ["100k samples/part.", "2.5m samples/part."]

    for row_idx, row_val in enumerate(row_vals):
        for col_idx, column_val in enumerate(column_vals):
            ax = axes[row_idx][col_idx]

            plot_baravg(all_data, ax, "0", column_val, [4], row_val) # TODO replace 4 workers with list of combinations

    HATCH_WIDTH()
    #FIG_LEGEND(fig)
    for row_idx, row in enumerate(axes):
        for col_idx, ax in enumerate(row):
            Y_GRID(ax)
            HIDE_BORDERS(ax)
            if row_idx == 0:
                ax.set_title(headings[col_idx])
            if col_idx == 0:
                ax.set_ylabel(f"{row_vals[row_idx]} trds", rotation=90, size='large')
    fig.supylabel('time (s)')
    fig.supxlabel('dataloader workers / prefetched partitions / parallel prefetch requests')

    fig.tight_layout()

    plot_path = os.path.join(plot_dir, "gridplot_threading")
    SAVE_PLOT(plot_path)
    PRINT_PLOT_PATHS()