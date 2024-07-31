import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from analytics.plotting.common.common import (
    hatch_width,
    hide_borders,
    init,
    load_data,
    print_plot_paths,
    save_plot,
    y_grid,
)


def plot_nbd(pipeline_log, ax, trigger):
    relevant_data = pipeline_log["supervisor"]["triggers"][trigger]["trainer_log"]
    all_epoch_timings = []
    for epoch in relevant_data["epochs"]:
        all_epoch_timings.extend(epoch["BatchTimings"])
    all_epoch_timings = np.array(all_epoch_timings) / 1000  # ms to seconds

    sns.histplot(data=all_epoch_timings, ax=ax, log_scale=True)

    # ax.set_xticks(list(x))
    # ax.set_xticklabels([f"{idx + 1}" for idx, _ in enumerate(x)])
    # ax.set_xlabel("Waiting time for next batch (seconds)")

    # ax.set_ylabel("Count")

    # ax.set_title("Histogram of waiting times")


def load_all_pipelines(data_path, worker_count_filter):
    all_data = []
    uniq_prefetched_partitions = set()
    uniq_parallel_prefetch_requests = set()

    for filename in glob.iglob(data_path + "/**/*.log", recursive=True):
        data = load_data(filename)
        num_data_loaders = data["configuration"]["pipeline_config"]["training"]["dataloader_workers"]
        prefetched_partitions = data["configuration"]["pipeline_config"]["training"]["num_prefetched_partitions"]
        parallel_prefetch_requests = data["configuration"]["pipeline_config"]["training"]["parallel_prefetch_requests"]

        if num_data_loaders == worker_count_filter:
            all_data.append(data)
            uniq_prefetched_partitions.add(prefetched_partitions)
            uniq_parallel_prefetch_requests.add(parallel_prefetch_requests)

    return (
        all_data,
        (len(uniq_prefetched_partitions), len(uniq_parallel_prefetch_requests)),
        uniq_prefetched_partitions,
        uniq_parallel_prefetch_requests,
    )


if __name__ == "__main__":
    data_path, plot_dir = init(sys.argv)
    WORKER_COUNT = 8

    (
        all_data,
        figure_dimensions,
        uniq_prefetched_partitions,
        uniq_parallel_prefetch_requests,
    ) = load_all_pipelines(data_path, WORKER_COUNT)

    fig, axes = plt.subplots(*figure_dimensions, figsize=(40, 20), sharex=True)

    row_vals = sorted(uniq_prefetched_partitions)
    column_vals = sorted(uniq_parallel_prefetch_requests)

    for row_idx, row_val in enumerate(row_vals):
        for col_idx, column_val in enumerate(column_vals):
            ax = axes[row_idx][col_idx]
            if row_idx == 0:
                ax.set_title(f"{column_val} PPR")
            if col_idx == 0:
                ax.set_ylabel(f"{row_val} PP", rotation=90, size="large")

            for data in all_data:
                prefetched_partitions = data["configuration"]["pipeline_config"]["training"][
                    "num_prefetched_partitions"
                ]
                parallel_prefetch_requests = data["configuration"]["pipeline_config"]["training"][
                    "parallel_prefetch_requests"
                ]

                if row_val == prefetched_partitions and column_val == parallel_prefetch_requests:
                    plot_nbd(data, ax, "0")

    hatch_width()
    # FIG_LEGEND(fig)
    for row in axes:
        for ax in row:
            y_grid(ax)
            hide_borders(ax)

    fig.tight_layout()

    plot_path = os.path.join(plot_dir, "next_batch_distribution")
    save_plot(plot_path)
    print_plot_paths()
