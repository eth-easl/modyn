from typing import Any

import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# Create the heatmap
from analytics.plotting.common.common import init_plot
from analytics.plotting.common.const import DOUBLE_FIG_HEIGHT, DOUBLE_FIG_WIDTH
from analytics.plotting.common.font import setup_font


def build_heatmap(
    heatmap_data: pd.DataFrame,
    y_ticks: list[int] | None = None,
    y_ticks_bins: int | None = None,
    x_ticks: list[int] | None = None,
    reverse_col: bool = False,
    y_label: str = "Reference Year",
    x_label: str = "Current Year",
    color_label: str = "Accuracy %",
    title_label: str = "",
    target_ax: Axes | None = None,
    height_factor: float = 1.0,
    width_factor: float = 1.0,
    square: bool = False,
    cbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    policy: list[tuple[int, int, int]] = [],
    cmap: Any | None = None,
    linewidth: int = 2,
) -> Figure | Axes:
    init_plot()
    setup_font(small_label=True, small_title=True)

    if not target_ax:
        fig = plt.figure(
            edgecolor="black",
            frameon=True,
            figsize=(
                DOUBLE_FIG_WIDTH * width_factor,
                2 * DOUBLE_FIG_HEIGHT * height_factor,
            ),
            dpi=300,
        )

    ax = sns.heatmap(
        heatmap_data,
        cmap=("RdBu" + ("_r" if reverse_col else "")) if not cmap else cmap,
        linewidths=0.0,
        linecolor="black",
        # color bar from 0 to 1
        cbar_kws={
            "label": color_label,
            # "ticks": [0, 25, 50, 75, 100],
            "orientation": "vertical",
        },
        ax=target_ax,
        square=square,
        **{
            "vmin": vmin if vmin is not None else heatmap_data.min().min(),
            "vmax": vmax if vmax is not None else heatmap_data.max().max(),
            "cbar": cbar,
        },
    )

    # Rasterize the heatmap background to avoid anti-aliasing artifacts
    ax.collections[0].set_rasterized(True)

    rect = patches.Rectangle(
        (0, 0),
        heatmap_data.shape[1],
        heatmap_data.shape[0],
        linewidth=2,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(rect)

    # Adjust x-axis tick labels
    ax.set_xlabel(x_label)
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

    ax.set_ylabel(y_label)

    if title_label:
        ax.set_title(title_label)

        # drift_pipeline = []

    # TODO visualize policy
    # Draft training boxes
    # if drift_pipeline:
    # x_start = active_[1][f"_start"].year - 1930
    # x_end = active_[1][f"{type_}_end"].year - 1930
    # y = active_[1]["model_idx"]

    previous_y = 0
    for x_start, x_end, y in policy:
        # main box
        rect = plt.Rectangle(
            (x_start, y),  # y: 0 based index, model_idx: 1 based index
            x_end - x_start,
            1,
            edgecolor="White",
            facecolor="none",
            linewidth=linewidth,
            linestyle="solid",
            hatch="/",
            joinstyle="bevel",
            # capstyle="round",
        )
        ax.add_patch(rect)

        # connector
        connector = plt.Rectangle(
            (x_start, previous_y),  # y: 0 based index, model_idx: 1 based index
            0,
            y - previous_y + 1,
            edgecolor="White",
            facecolor="none",
            linewidth=linewidth,
            linestyle="solid",
            hatch="/",
            joinstyle="bevel",
            # capstyle="round",
        )
        ax.add_patch(connector)
        previous_y = y

    # Display the plot
    plt.tight_layout()
    # plt.show()

    return fig if not target_ax else ax
