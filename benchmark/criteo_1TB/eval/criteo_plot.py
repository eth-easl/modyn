import sys

import matplotlib.pyplot as plt
from plotting.common import *


def plot_accuracy(pipelines_data, ax):
    for pipeline in sorted(pipelines_data):
        pipeline_data = pipelines_data[pipeline]

        x = pipeline_data.keys()
        y = pipeline_data.values()

        y_rounded = [round(y["auc"], 2) for y in y]
        ax.plot(x, y_rounded, **LINE(pipeline), label=PIPELINE_NAME[pipeline])

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{idx + 1}" for idx, _ in enumerate(x)])
    ax.set_xlabel("Day (Trigger)")

    ax.set_ylabel("AUC")
    ax.set_ylim(0.7, 1.0)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])

    ax.set_title("Model Performance on Day 10")


if __name__ == "__main__":
    data_path, plot_dir = INIT(sys.argv)
    data = LOAD_DATA(data_path)

    fig, ax = plt.subplots(1, 1, figsize=DOUBLE_FIG_SIZE)

    plot_accuracy(data, ax)

    HATCH_WIDTH()
    # FIG_LEGEND(fig)
    Y_GRID(ax)
    HIDE_BORDERS(ax)

    plot_path = os.path.join(plot_dir, "criteo_auc")
    SAVE_PLOT(plot_path)
    PRINT_PLOT_PATHS()
