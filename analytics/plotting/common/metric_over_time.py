import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analytics.plotting.common.color import main_color
from analytics.plotting.common.common import (
    DOUBLE_FIG_HEIGHT,
    DOUBLE_FIG_WIDTH,
    init_plot,
)
from analytics.plotting.common.font import setup_font


def plot_metric_over_time(
    data: pd.DataFrame,
    x: str = "time",
    y: str = "value",
    hue: str = "pipeline_ref",
    style: str = "pipeline_ref",
    y_label: str = "Reference Year",
    x_label: str = "Current Year",
    title_label: str = "",
    target_ax: Axes | None = None,
    height_factor: float = 1.0,
    width_factor: float = 1.0,
    grid_alpha: float = 0.0,
    legend_label: str = "TODO",
    small_legend_fonts: bool = False,
    x_date_locator: mdates.DateLocator | None = None,
    x_date_formatter: mdates.DateFormatter | None = None,
    y_ticks: list[int] | None = None,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[int, int] | None = None,
    markers: bool = True,
) -> Figure | Axes:
    sns.set_style("whitegrid")
    init_plot()
    setup_font(small_label=False, small_title=False, small_ticks=False)

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

    ax = sns.lineplot(
        data,
        x=x,
        y=y,
        hue=hue,
        markersize=7,
        # line width
        linewidth=2.5,
        palette=[
            main_color(0),
            main_color(1),
            main_color(3),
            main_color(4),
            main_color(5),
            main_color(6),
        ],
        style=style,
        markers=markers,
    )

    if xlim:
        ax.set(xlim=xlim)

    if ylim:
        ax.set(ylim=ylim)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.legend(
        title=legend_label,
        ncol=2,
        handletextpad=1,
        columnspacing=1.4,
        **({"fontsize": "x-small"} if small_legend_fonts else {}),
    )

    if x_date_locator:
        ax.xaxis.set_major_locator(x_date_locator)
        # ax.set_xticklabels(x_ticks, rotation=0)
        ax.xaxis.set_major_formatter(x_date_formatter)
        # ticks = ax.get_xticks()
        plt.xticks(rotation=0)

    if y_ticks:
        ax.set_yticks(y_ticks)

    # Display the plot
    plt.tight_layout()

    return fig if not target_ax else ax
