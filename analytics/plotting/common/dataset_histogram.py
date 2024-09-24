import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from analytics.plotting.common.color import (
    gen_categorical_map,
    main_color,
    main_colors,
)
from analytics.plotting.common.common import DOUBLE_FIG_HEIGHT, init_plot
from analytics.plotting.common.const import DOUBLE_FIG_WIDTH
from analytics.plotting.common.font import setup_font


def build_countplot(
    histogram_data: pd.DataFrame,
    x: str,
    y_ticks: list[int] | None = None,
    y_ticks_bins: int | None = None,
    x_ticks: list[int] | None = None,
    y_label: str = "Number of Samples",
    x_label: str = "Year",
    height_factor: float = 1.0,
    width_factor: float = 1.0,
    palette: str = "RdBu",
    palette_strip: tuple[float, float] | None = (0.35, 0.65),
) -> Figure:
    init_plot()
    setup_font()

    fig = plt.figure(
        edgecolor="black",
        frameon=True,
        figsize=(
            DOUBLE_FIG_WIDTH * width_factor,
            2 * DOUBLE_FIG_HEIGHT * height_factor,
        ),
        dpi=600,
    )
    ax = fig.add_subplot(111)

    agg_by_year = histogram_data.groupby(x).size().reset_index(name="count")

    ax = sns.barplot(
        data=agg_by_year,
        x=x,
        y="count",
        color=main_color(0),
        # hue="count",
        # palette=get_rdbu_wo_white(palette=palette, strip=palette_strip),
        width=1,
        legend=False,
        # bins=12,
        # element="step",  # hide lines
        # native_scale=True,        ax=ax,
    )

    # avoid fine white lines between cells
    for artist in ax.patches:  # ax.patches contains the bars in the plot
        artist.set_rasterized(True)

    # draw grid behind bars (horizontal and vertical)
    ax.grid(axis="x", linestyle="--", alpha=1.0)
    ax.grid(axis="y", linestyle="--", alpha=1.0)

    # Adjust x-axis tick labels
    plt.xlabel(x_label)
    if x_ticks is not None:
        plt.xticks(
            ticks=[xtick - min(histogram_data[x]) for xtick in x_ticks],
            labels=x_ticks,
            rotation=0,
            # ha='right'
        )

    plt.ylabel(y_label)
    if y_ticks is not None:
        plt.yticks(ticks=y_ticks, labels=y_ticks, rotation=0)
    elif y_ticks_bins is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks_bins))
        ax.set_yticklabels([int(i) for i in ax.get_yticks()], rotation=0)

    # Display the plot
    plt.tight_layout()
    # plt.show()

    return fig


def build_histogram_multicategory_facets(
    histogram_data: pd.DataFrame,
    x: str,
    label: str,
    sorted_categories: pd.Series,
    y_ticks: list[int | float] | None = None,
    y_ticks_bins: int | None = None,
    x_ticks: list[pd.Timestamp] | None = None,
    y_label: str = "Number of Samples",
    x_label: str = "Year",
    sharey: bool = False,
    height_factor: float = 1.0,
    width_factor: float = 1.0,
    legend_labels: list[str] | None = None,
) -> Figure:
    color_map = gen_categorical_map(sorted_categories)
    histogram_data = histogram_data.copy()

    init_plot()
    setup_font()

    # Create a FacetGrid object with 'sex' as the categorical label for facets
    g = sns.FacetGrid(
        histogram_data,
        col=label,
        margin_titles=False,
        col_wrap=6,
        sharey=sharey,  # sharey=False allows independent y-axis
        sharex=True,
        col_order=sorted_categories,
        subplot_kws={},
        despine=True,
        # gridspec_kws={"hspace": 0, "wspace": 0},
    )

    g.figure.set_dpi(300)
    g.figure.set_figwidth(DOUBLE_FIG_WIDTH * width_factor)
    g.figure.set_figheight(2 * DOUBLE_FIG_HEIGHT * height_factor)

    g.map_dataframe(
        sns.histplot,
        # data=histogram_data, # supplied by map_dataframe
        x=x,
        hue=label,
        palette=color_map,
        edgecolor=None,  # Disable black borders
        element="bars",  # bars, poly, bars
        multiple="dodge",  # layer, **dodge**, **fill**, **stack**
        bins=40,
    )

    g.set_titles("{col_name}")  # only the value in the facet name

    # Adjust x-axis tick labels
    # g.set(xlabel=x_label)
    if x_ticks is not None:
        g.set(xticks=x_ticks)

        for ax in g.axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
            ax.figure.autofmt_xdate(ha="center", rotation=0)  # Auto-rotate the date labels

    for ax in g.axes.flat:
        # draw grid behind bars (horizontal and vertical)
        ax.grid(axis="x", alpha=1.0, linestyle="--")
        ax.grid(axis="y", alpha=1.0, linestyle="--")

    # g.set(ylabel=y_label)
    # Hide y-axis labels for all but the leftmost column
    for i, ax in enumerate(g.axes.flat):
        # ax.set_xlabel(x_label, labelpad=10)
        # if i % 4 != 0:  # Check if it's not in the leftmost column
        ax.set_ylabel(None)
        ax.set_xlabel(None)

        # center the x-axis labels
        ax.tick_params(axis="x", rotation=0, pad=6)
        ax.tick_params(axis="y", pad=10)

        # avoid fine white lines between cells
        for artist in ax.patches:  # ax.patches contains the bars in the plot
            artist.set_rasterized(True)

    # g.set_axis_labels(
    #     x_var=x_label,
    #     y_var=y_label,
    #     clear_inner=True,
    # )

    # Add common x and y labels with custom placement
    g.figure.text(0.5, 0.0, x_label, ha="center", va="center", fontsize="large")
    g.figure.text(
        0.0,
        0.5,
        y_label,
        ha="center",
        va="center",
        rotation="vertical",
        fontsize="large",
    )

    plt.tight_layout()
    g.figure.subplots_adjust(wspace=0.4)  # Reduce horizontal space between subplots
    return g


def build_histogram_multicategory_barnorm(
    histogram_data: pd.DataFrame,
    x: str,
    label: str,
    sorted_coloring_categories: pd.Series,
    sorted_ordering_categories: pd.Series | None = None,
    y_ticks: list[int | float] | None = None,
    y_ticks_bins: int | None = None,
    x_ticks: list[pd.Timestamp] | None = None,
    y_label: str = "Number of Samples",
    x_label: str = "Year",
    height_factor: float = 1.0,
    width_factor: float = 1.0,
    legend: bool = True,
    legend_labels: list[str] | None = None,
    legend_title: str | None = None,
    nbins: int | None = None,
    manual_color_map: dict[str, tuple[float, float, float]] | None = None,
    grid_opacity: float = 1.0,
    col_alpha: float | None = None,
) -> Figure:
    if sorted_ordering_categories is None:
        sorted_ordering_categories = sorted_coloring_categories
    if legend_labels is None:
        legend_labels = []

    histogram_data = histogram_data.copy()
    # rename: if label not in legend_labels, add underscore to label to hide it from the legend
    histogram_data[label] = histogram_data[label].apply(lambda x: x if x in (legend_labels) else f"_{x}")
    underscore_col_categories = [x if x in (legend_labels) else f"_{x}" for x in sorted_coloring_categories]
    underscore_ordering_categories = [x if x in (legend_labels) else f"_{x}" for x in sorted_ordering_categories]
    underscore_ordering_categories += [
        # add any missing categories to the end of the list
        x
        for x in underscore_col_categories
        if x not in underscore_ordering_categories
    ]
    color_map = gen_categorical_map(underscore_col_categories)

    init_plot()
    setup_font()

    fig = plt.figure(
        edgecolor="black",
        frameon=True,
        figsize=(
            DOUBLE_FIG_WIDTH * width_factor,
            2 * DOUBLE_FIG_HEIGHT * height_factor,
        ),
        dpi=600,
    )
    ax = fig.add_subplot(111)

    ax = sns.histplot(
        data=histogram_data,
        x=x,
        hue=label,
        palette=manual_color_map if manual_color_map else color_map,
        hue_order=underscore_ordering_categories,
        linewidth=0,  # avoid fine white lines between cells
        edgecolor=None,  # Disable black borders
        # legend=len(legend_labels or []) > 0,
        legend=legend,
        element="bars",  # bars, poly, bars
        multiple="fill",  # layer, **dodge**, **fill**, **stack**
        **{"bins": nbins} if nbins is not None else {},
        # opacity
        **{"alpha": col_alpha} if col_alpha is not None else {},
        ax=ax,
    )
    ax.invert_yaxis()

    # avoid fine white lines between cells
    for artist in ax.patches:  # ax.patches contains the bars in the plot
        artist.set_rasterized(True)

    # position legend outside of plot
    if legend and len(legend_labels) > 0:
        ax.get_legend().set_bbox_to_anchor((1.05, 1.05))

        if legend_title is not None:
            ax.get_legend().set_title(legend_title)

    # draw grid behind bars (horizontal and vertical)
    ax.grid(axis="x", linestyle="--", alpha=grid_opacity, color="white")
    ax.grid(axis="y", linestyle="--", alpha=grid_opacity, color="white")

    # Adjust x-axis tick labels
    plt.xlabel(x_label)
    if x_ticks is not None:
        # ax.xaxis.set_major_locator(DateLocator())
        # ax.set_xticklabels(x_ticks, rotation=0)
        # plt.xticks(
        #     ticks=x_ticks,
        #     labels=x_ticks,
        #     rotation=0,
        #     # ha='right'
        # )
        plt.xticks(x_ticks)
        date_form = mdates.DateFormatter("%b\n%Y")  # Customize format: "2020 Jan"
        ax.xaxis.set_major_formatter(date_form)

        # Optionally, adjust the number of ticks on x-axis
        # ax.xaxis.set_major_locator(mdates.YearLocator(base=4))  # Show every 3 months

    # ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks_bins))
    # # ax.set_yticklabels([int(i) + histogram_data["x"].min() for i in ax.get_yticks()], rotation=0)

    plt.ylabel(y_label)
    if y_ticks is not None:
        plt.yticks(ticks=y_ticks, labels=list(reversed(y_ticks)), rotation=0)
    elif y_ticks_bins is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks_bins))
        # ax.set_yticklabels([int(i) for i in ax.get_yticks()], rotation=0)

    # Display the plot
    plt.tight_layout()
    # plt.show()

    return fig


def build_cum_barplot(
    histogram_data: pd.DataFrame,
    x: str,
    y: str,
    y_ticks: list[int] | None = None,
    y_ticks_bins: int | None = None,
    x_ticks: list[int] | None = None,
    x_ticks_bins: int | None = None,
    y_label: str = "Number of Samples",
    x_label: str = "Year",
    height_factor: float = 1.0,
    width_factor: float = 1.0,
    palette: str = "RdBu",
    palette_strip: tuple[float, float] | None = (0.35, 0.65),
) -> Figure:
    init_plot()
    setup_font()

    fig = plt.figure(
        edgecolor="black",
        frameon=True,
        figsize=(
            DOUBLE_FIG_WIDTH * width_factor,
            2 * DOUBLE_FIG_HEIGHT * height_factor,
        ),
        dpi=600,
    )
    ax = fig.add_subplot(111)

    ax = sns.lineplot(
        data=histogram_data,
        x=x,
        y=y,
        color=main_color(0),
        # market size
        # markers=False
        # markers=True,
        # hue=y,
        # # palette=get_rdbu_wo_white(palette=palette, strip=palette_strip),
        # width=1,
        # legend=False,
        # # fill=True,
        # edgecolor=".5",
        # facecolor=(0, 0, 0, 0),
        ax=ax,
    )
    # TODO: check gap, dodged elements --> if pdf shows white lines

    # draw grid behind bars (horizontal and vertical)
    ax.grid(axis="x", linestyle="--", alpha=1.0)
    ax.grid(axis="y", linestyle="--", alpha=1.0)

    # Adjust x-axis tick labels
    plt.xlabel(x_label)
    if x_ticks is not None:
        plt.xticks(
            ticks=[xtick - min(histogram_data[x]) for xtick in x_ticks],
            labels=x_ticks,
            rotation=0,
            # ha='right'
        )
    elif x_ticks_bins is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=x_ticks_bins))

    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks_bins))
    # ax.set_yticklabels([int(i) + histogram_data["x"].min() for i in ax.get_yticks()], rotation=0)

    plt.ylabel(y_label)
    if y_ticks is not None:
        plt.yticks(ticks=y_ticks, labels=y_ticks, rotation=0)
    elif y_ticks_bins is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks_bins))
        # ax.set_yticklabels([int(i) for i in ax.get_yticks()], rotation=0)

    # Display the plot
    plt.tight_layout()
    # plt.show()

    return fig


def build_pieplot(
    x: list[int],
    labels: list[str],
    height_factor: float = 1.0,
    width_factor: float = 1.0,
) -> Figure:
    init_plot()
    setup_font()

    fig = plt.figure(
        edgecolor="black",
        frameon=True,
        figsize=(
            DOUBLE_FIG_WIDTH * width_factor,
            2 * DOUBLE_FIG_HEIGHT * height_factor,
        ),
        dpi=600,
    )

    def func(pct, allvals):
        absolute = int(np.round(pct / 100.0 * np.sum(allvals)))
        return f"{pct:.1f}%\n({absolute:d})"

    wedges, texts, autotexts = plt.pie(
        x=x,
        labels=labels,
        autopct=lambda pct: func(pct, x),
        textprops=dict(color="w"),
        colors=main_colors(),
        # show labels next to the pie chart
        startangle=90,
        explode=(0.1, 0),
    )

    plt.setp(autotexts, size=8, weight="bold")

    # Display the plot
    plt.tight_layout()
    # plt.show()

    return fig
