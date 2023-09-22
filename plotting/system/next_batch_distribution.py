import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from plotting.common.common import *


def plot_nbd(pipeline_log, ax, trigger):
    relevant_data = pipeline_log["supervisor"]["triggers"][trigger]["trainer_log"]
    all_epoch_timings = []
    for epoch in relevant_data["epochs"]:
        all_epoch_timings.extend(epoch["BatchTimings"])
    all_epoch_timings = np.array(all_epoch_timings) / 1000 # ms to seconds
    

    sns.histplot(data=all_epoch_timings, ax=ax, log_scale=True)
   
    #ax.set_xticks(list(x))
    #ax.set_xticklabels([f"{idx + 1}" for idx, _ in enumerate(x)])
    ax.set_xlabel("Waiting time for next batch (seconds)")

    ax.set_ylabel("Count")

    ax.set_title("Histogram of waiting times")


if __name__ == '__main__':
    data_path, plot_dir = INIT(sys.argv)
    data = LOAD_DATA(data_path)

    fig, ax = plt.subplots(1, 1, figsize=DOUBLE_FIG_SIZE)

    plot_nbd(data, ax, "0")

    HATCH_WIDTH()
    #FIG_LEGEND(fig)
    Y_GRID(ax)
    HIDE_BORDERS(ax)

    plot_path = os.path.join(plot_dir, "next_batch_distribution")
    SAVE_PLOT(plot_path)
    PRINT_PLOT_PATHS()