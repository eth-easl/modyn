import os
import sys

import matplotlib.pyplot as plt

from analytics.plotting.common.common import (
    DOUBLE_FIG_SIZE,
    PIPELINE_NAME,
    hatch_width,
    hide_borders,
    line,
    load_data,
    print_plot_paths,
    save_plot,
    y_grid,
)
from benchmark.criteo_1TB.eval.plotting.common import init


def plot_accuracy(pipelines_data, ax):
    for pipeline in sorted(pipelines_data):
        pipeline_data = pipelines_data[pipeline]

        x = pipeline_data.keys()
        y = pipeline_data.values()

        y_rounded = [round(y["auc"], 2) for y in y]
        ax.plot(x, y_rounded, **line(pipeline), label=PIPELINE_NAME[pipeline])

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{idx + 1}" for idx, _ in enumerate(x)])
    ax.set_xlabel("Day (Trigger)")

    ax.set_ylabel("AUC")
    ax.set_ylim(0.7, 1.0)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])

    ax.set_title("Model Performance on Day 10")


if __name__ == "__main__":
    data_path, plot_dir = init(sys.argv)
    data = load_data(data_path)

    fig, ax = plt.subplots(1, 1, figsize=DOUBLE_FIG_SIZE)

    plot_accuracy(data, ax)

    hatch_width()
    # FIG_LEGEND(fig)
    y_grid(ax)
    hide_borders(ax)

    plot_path = os.path.join(plot_dir, "criteo_auc")
    save_plot(plot_path)
    print_plot_paths()
