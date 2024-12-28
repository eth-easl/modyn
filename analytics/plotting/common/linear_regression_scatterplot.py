from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analytics.plotting.common.color import main_color
from analytics.plotting.common.common import init_plot
from analytics.plotting.common.font import setup_font


def scatter_linear_regression(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    y_ticks: list[int] | list[str] | None = None,
    x_ticks: list[int] | None = None,
    y_label: str = "Reference Year",
    x_label: str = "Current Year",
    height_factor: float = 1.0,
    width_factor: float = 1.0,
    legend_label: str = "Number Samples",
    title_label: str = "",
    target_ax: Axes | None = None,
    palette: Any = None,
    small_legend_fonts: bool = False,
) -> Figure | tuple[Axes, Axes]:
    sns.set_style("whitegrid")

    init_plot()
    setup_font(small_label=True, small_title=True)

    DOUBLE_FIG_WIDTH = 10
    DOUBLE_FIG_HEIGHT = 3.5

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

    ax1 = sns.regplot(
        data,
        x=x,
        y=y,  # duration
        color=main_color(0),
    )

    ax2 = sns.scatterplot(
        data,
        x=x,
        y=y,  # duration
        hue=hue,
        palette=palette,
        s=200,
        legend=True,
        marker="X",
    )

    ax2.legend(
        title=legend_label,
        ncol=2,
        handletextpad=0,
        columnspacing=0.5,
        **({"fontsize": "x-small"} if small_legend_fonts else {}),
    )

    # Adjust x-axis tick labels
    ax2.set_xlabel(x_label)
    if x_ticks is not None:
        ax2.set_xticks(
            ticks=x_ticks,
            labels=x_ticks,
            rotation=0,
            # ha='right'
        )

    if y_ticks is not None:
        ax2.set_yticks(
            ticks=y_ticks,
            labels=y_ticks,
            rotation=0,
        )

    ax2.set_ylabel(y_label)

    if title_label:
        ax2.set_title(title_label)

    print("Number of plotted items", data.shape[0])

    # Display the plot
    plt.tight_layout()

    return fig if not target_ax else (ax1, ax2)
