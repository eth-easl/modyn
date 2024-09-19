from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# Create the heatmap
from analytics.plotting.common.common import init_plot


def build_heatmap(
    heatmap_data: pd.DataFrame,
    y_ticks: list[int] | None = None,
    y_ticks_bins: int | None = None,
    x_ticks: list[int] | None = None,
    reverse_col: bool = False,
    y_label: str = "Reference Year",
    x_label: str = "Current Year",
    color_label: str = "Accuracy %",
    target_ax: Axes | None = None,
) -> Figure:
    init_plot()
    # sns.set_theme(style="ticks")
    plt.rcParams["svg.fonttype"] = "none"

    double_fig_width = 10
    double_fig_height = 3.5

    fig = plt.figure(
        edgecolor="black",
        frameon=True,
        figsize=(double_fig_width, 2.2 * double_fig_height),
        dpi=300,
    )

    ax = sns.heatmap(
        heatmap_data,
        cmap="RdBu" + ("_r" if reverse_col else ""),
        linewidths=0.0,
        linecolor="black",
        cbar=True,
        # color bar from 0 to 1
        cbar_kws={
            "label": color_label,
            # "ticks": [0, 25, 50, 75, 100],
            "orientation": "vertical",
        },
        # TODO
        ax=target_ax,
    )
    ax.collections[0].set_rasterized(True)

    # Adjust x-axis tick labels
    plt.xlabel(x_label)
    if not x_ticks:
        ax.set_xticks(
            ticks=[x + 0.5 for x in range(0, 2010 - 1930 + 1, 20)],
            labels=[x for x in range(1930, 2010 + 1, 20)],
            rotation=0,
            # ha='right'
        )
    else:
        ax.set_xticks(
            ticks=[x - 1930 + 0.5 for x in x_ticks],
            labels=[x for x in x_ticks],
            rotation=0,
            # ha='right'
        )
    ax.invert_yaxis()

    if y_ticks is not None:
        ax.set_yticks(
            ticks=[y + 0.5 - 1930 for y in y_ticks],
            labels=[y for y in y_ticks],
            rotation=0,
        )
    elif y_ticks_bins is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks_bins))
        ax.set_yticklabels([int(i) + min(heatmap_data.index) for i in ax.get_yticks()], rotation=0)

    plt.ylabel(y_label)

    # # Draft training boxes
    # if drift_pipeline:
    #     for type_, dashed in [("train", False), ("usage", False), ("train", True)]:
    #         for active_ in df_logs_models.iterrows():
    #             x_start = active_[1][f"{type_}_start"].year - 1930
    #             x_end = active_[1][f"{type_}_end"].year - 1930
    #             y = active_[1]["model_idx"]
    #             rect = plt.Rectangle(
    #                 (x_start, y - 1),  # y: 0 based index, model_idx: 1 based index
    #                 x_end - x_start,
    #                 1,
    #                 edgecolor="White" if type_ == "train" else "Black",
    #                 facecolor="none",
    #                 linewidth=3,
    #                 linestyle="dotted" if dashed else "solid",
    #                 hatch="/",
    #                 joinstyle="bevel",
    #                 # capstyle="round",
    #             )
    #             ax.add_patch(rect)

    # Display the plot
    plt.tight_layout()
    # plt.show()

    return fig


def save_plot(fig: Figure, name: str) -> None:
    for img_type in ["png", "svg"]:
        img_path = Path("/scratch/robinholzi/gh/modyn/.data/plots") / f"{name}.{img_type}"
        fig.savefig(img_path, bbox_inches="tight", transparent=True)
