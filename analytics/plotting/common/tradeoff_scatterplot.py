import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analytics.plotting.common.color import main_color
from analytics.plotting.common.common import init_plot
from analytics.plotting.common.font import setup_font


def plot_tradeoff_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    style: str,
    x_label: str = "Number of Triggers",
    y_label: str = "Mean Accuracy %",
    height_factor: float = 1.0,
    width_factor: float = 1.0,
    target_ax: Axes | None = None,
    manual_legend_title: bool = True,
    legend_ncol: int = 1,
) -> Figure:
    sns.set_theme(style="whitegrid")
    init_plot()
    setup_font(small_label=True, small_title=True, small_ticks=True)

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

    ax = sns.scatterplot(
        data,
        x=x,
        y=y,
        hue=hue,
        style=style,
        palette=[
            main_color(0),
            main_color(1),
            main_color(3),
            main_color(4),
            main_color(5),
        ],
        s=300,
        # legend=False,
        # marker="X",
    )

    ax.legend(
        fontsize="small",
        title_fontsize="medium",
        # title="Pipeline",
        **(
            {
                "title": hue,
            }
            if manual_legend_title
            else {}
        ),
        # 2 columns
        ncol=legend_ncol,
    )

    # Adjust x-axis tick labels
    plt.xlabel(x_label, labelpad=10)

    # Set y-axis ticks to be equally spaced
    plt.ylabel(y_label, labelpad=15)

    # Display the plot
    plt.tight_layout()
    plt.show()

    return fig
