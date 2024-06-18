# Credits to Lawrence Benson (https://github.com/hpides/perma-bench/tree/eval/scripts)

import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt

#######################################
# Plotting
#######################################

FS = 20
MILLION = 1_000_000
SINGLE_FIG_WIDTH = 5
SINGLE_FIG_HEIGHT = 3.5
SINGLE_FIG_SIZE = (SINGLE_FIG_WIDTH, SINGLE_FIG_HEIGHT)
DOUBLE_FIG_WIDTH = 10
DOUBLE_FIG_HEIGHT = 3.5
DOUBLE_FIG_SIZE = (DOUBLE_FIG_WIDTH, DOUBLE_FIG_HEIGHT)
PLOT_PATHS = []
IMG_TYPES = [".png"]  # add .svg here to generate svg

PIPELINE_COLOR = {
    "models_exp0_finetune": "#a1dab4",
    "retrain_noreset": "#378d54",
    "apache-512": "#41b6c4",
    "barlow-256": "#2c7fb8",
    "barlow-512": "#2c7fb8",
    "z-barlow-dram": "#253494",
    "z-apache-dram": "#0c1652",
}

PIPELINE_MARKER = {
    "models_exp0_finetune": "P",
    "retrain_noreset": "o",
    "apache-512": "d",
    "barlow-256": "s",
    "barlow-512": ".",
    "z-apache-dram": "x",
    "z-barlow-dram": "^",
}

PIPELINE_HATCH = {
    "models_exp0_finetune": "\\\\",
    "retrain_noreset": "//",
    "apache-512": "\\",
    "barlow-256": "/",
    "barlow-512": ".",
    "z-apache-dram": ".",
    "z-barlow-dram": "x",
}

PIPELINE_NAME = {
    "models_exp0_finetune": "Finetuning",
    "retrain_noreset": "Retrain",
    "apache-512": "A-512",
    "barlow-256": "B-256",
    "barlow-512": "B-256-PF",
    "z-apache-dram": "A-D",
    "z-barlow-dram": "B-D",
}


def INIT_PLOT():
    matplotlib.rcParams.update(
        {
            "font.size": FS,
            "svg.fonttype": "none",
        }
    )


def PRINT_PLOT_PATHS():
    print(f"To view new plots, run:\n\topen {' '.join(PLOT_PATHS)}")


def BAR(system):
    return {"color": "white", "edgecolor": PIPELINE_COLOR[system], "hatch": PIPELINE_HATCH[system], "lw": 3}


def LINE(system):
    return {
        "lw": 4,
        "ms": 10,
        "color": PIPELINE_COLOR[system],
        "marker": PIPELINE_MARKER[system],
        "markeredgewidth": 1,
        "markeredgecolor": "black",
    }


def BAR_X_TICKS_POS(bar_width, num_bars, num_xticks):
    return [i - (bar_width / 2) + ((num_bars * bar_width) / 2) for i in range(num_xticks)]


def RESIZE_TICKS(ax, x=FS, y=FS):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(x)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(y)


def HATCH_WIDTH(width=4):
    matplotlib.rcParams["hatch.linewidth"] = width


def Y_GRID(ax):
    ax.grid(axis="y", which="major")
    ax.set_axisbelow(True)


def HIDE_BORDERS(ax, show_left=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(show_left)


def FIG_LEGEND(fig):
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=6,
        frameon=False,
        columnspacing=1,
        handletextpad=0.3,
        # , borderpad=0.1, labelspacing=0.1, handlelength=1.8
    )
    fig.tight_layout()


def LOAD_DATA(path):
    with open(path) as json_file:
        return json.load(json_file)


def SAVE_PLOT(plot_path, img_types=None):
    if img_types is None:
        img_types = IMG_TYPES

    for img_type in img_types:
        img_path = f"{plot_path}{img_type}"
        PLOT_PATHS.append(img_path)
        plt.savefig(img_path, bbox_inches="tight", dpi=300)

    plt.figure()


def INIT(args):
    if len(args) != 3:
        sys.exit("Need /path/to/results /path/to/plots")

    result_path = args[1]
    plot_dir = args[2]

    os.makedirs(plot_dir, exist_ok=True)
    INIT_PLOT()

    return result_path, plot_dir
