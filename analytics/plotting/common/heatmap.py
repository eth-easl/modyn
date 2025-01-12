from typing import Any, Literal, cast

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


def get_fractional_index(dates: pd.Index, query_date: pd.Timestamp, fractional: bool = True) -> float:
    """Given a list of Period objects (dates) and a query_date as a Period,
    return the interpolated fractional index between two period indices if the
    query_date lies between them."""
    # Ensure query_date is within the bounds of the period range
    if query_date < dates[0].start_time:
        return -1  # -1 before first index

    if query_date > dates[-1].start_time:
        return len(dates)  # +1 after last index

    # Find the two periods where the query_date falls in between
    for i in range(len(dates) - 1):
        if dates[i].start_time <= query_date <= dates[i + 1].start_time:
            # Perform linear interpolation, assuming equal length periods
            return i + (
                ((query_date - dates[i].start_time) / (dates[i + 1].start_time - dates[i].start_time))
                if fractional
                else 0
            )

    # If query_date is exactly one of the dates
    return dates.get_loc(query_date)


def build_heatmap(
    heatmap_data: pd.DataFrame,
    y_ticks: list[int] | list[str] | None = None,
    y_ticks_bins: int | None = None,
    x_ticks: list[int] | None = None,
    x_custom_ticks: list[tuple[int, str]] | None = None,  # (position, label)
    y_custom_ticks: list[tuple[int, str]] | None = None,  # (position, label)
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
    grid_alpha: float = 0.0,
    disable_horizontal_grid: bool = False,
    df_logs_models: pd.DataFrame | None = None,
    triggers: dict[int, pd.DataFrame] = {},
    x_axis: Literal["year", "other"] = "year",
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
            dpi=600,
        )

    ax = sns.heatmap(
        heatmap_data,
        cmap=("RdBu" + ("_r" if reverse_col else "")) if not cmap else cmap,
        linewidths=0.0,
        linecolor="white",
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
    if not x_ticks and not x_custom_ticks:
        ax.set_xticks(
            ticks=[x + 0.5 for x in range(0, 2010 - 1930 + 1, 20)],
            labels=[x for x in range(1930, 2010 + 1, 20)],
            rotation=0,
            # ha='right'
        )
    else:
        if x_custom_ticks:
            ax.set_xticks(
                ticks=[x[0] for x in x_custom_ticks],
                labels=[x[1] for x in x_custom_ticks],
                rotation=0,
                # ha='right'
            )
        else:
            assert x_ticks is not None
            ax.set_xticks(
                ticks=[x - 1930 + 0.5 for x in x_ticks],
                labels=[x for x in x_ticks],
                rotation=0,
                # ha='right'
            )
    ax.invert_yaxis()

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0 if disable_horizontal_grid else grid_alpha,
        color="white",
    )
    ax.grid(axis="x", linestyle="--", alpha=grid_alpha, color="white")

    if y_ticks is not None:
        ax.set_yticks(
            ticks=[int(y) + 0.5 - 1930 for y in y_ticks],
            labels=[y for y in y_ticks],
            rotation=0,
        )
    elif y_ticks_bins is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks_bins))
        ax.set_yticklabels([int(i) + min(heatmap_data.index) for i in ax.get_yticks()], rotation=0)
    else:
        if y_custom_ticks:
            ax.set_yticks(
                ticks=[y[0] for y in y_custom_ticks],
                labels=[y[1] for y in y_custom_ticks],
                rotation=0,
                # ha='right'
            )

    ax.set_ylabel(y_label)

    if title_label:
        ax.set_title(title_label)

    # mainly for offline expore
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

    # for post factum evaluation
    if df_logs_models is not None:
        for type_, dashed in [("train", False), ("usage", False), ("train", True)]:
            for active_ in df_logs_models.iterrows():
                if x_axis == "year":
                    eval_x_start = active_[1][f"{type_}_start"].year - 1930
                    eval_x_end = active_[1][f"{type_}_end"].year - 1930
                else:
                    eval_x_start = get_fractional_index(
                        heatmap_data.columns,
                        cast(pd.Index, active_[1][f"{type_}_start"]),
                        fractional=False,
                    )
                    eval_x_end = get_fractional_index(
                        heatmap_data.columns,
                        cast(pd.Index, active_[1][f"{type_}_end"]),
                        fractional=False,
                    )

                y = active_[1]["model_idx"]
                rect = plt.Rectangle(
                    (
                        eval_x_start,
                        y - 1,
                    ),  # y: 0 based index, model_idx: 1 based index
                    eval_x_end - eval_x_start,
                    1,
                    edgecolor="White" if type_ == "train" else "Black",
                    facecolor="none",
                    linewidth=1.5,
                    linestyle="dotted" if dashed else "solid",
                    hatch="/",
                    joinstyle="bevel",
                    # capstyle="round",
                )
                ax.add_patch(rect)

    if triggers:
        for y, triggers_df in triggers.items():
            for row in triggers_df.iterrows():
                type_ = "usage"
                # for y, x_list in triggers.items():
                eval_x_start = row[1][f"{type_}_start"].year - 1930
                eval_x_end = row[1][f"{type_}_end"].year - 1930
                # for x in x_list:
                rect = plt.Rectangle(
                    (eval_x_start, y),  # y: 0 based index, model_idx: 1 based index
                    eval_x_end - eval_x_start,
                    1,
                    edgecolor="black",
                    facecolor="none",
                    linewidth=1,
                    # linestyle="dotted",
                    # hatch="/",
                    # joinstyle="bevel",
                    # capstyle="round",
                )
                ax.add_patch(rect)

    # Display the plot
    plt.tight_layout()
    # plt.show()

    return fig if not target_ax else ax
