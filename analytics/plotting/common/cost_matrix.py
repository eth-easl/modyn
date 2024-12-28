import matplotlib.dates as mdates
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


def plot_cost_matrix(
    df_costs: pd.DataFrame,
    pipeline_ids: list[int],
    grid_alpha: float = 0.0,
    title_map: dict[int, str] = {},
    height_factor: float = 1.0,
    width_factor: float = 1.0,
    duration_ylabel: str = "Duration (sec.)",
    cumulative_ylabel: str = "Cumulative Duration (sec.)",
    x_label: str = "Sample Timestamp",
    x_lim: tuple[int, int] = (1930, 2013),
    x_ticks: list[int] | None = None,
    x_date_locator: mdates.DateLocator | None = None,
    x_date_formatter: mdates.DateFormatter | None = None,
    y_lim: tuple[int, int] = (0, 4000),
    y_lim_cumulative: tuple[int, int] = (0, 4000),
    y_ticks: list[int] | None = None,
    y_ticks_cumulative: list[int] | None = None,
    y_minutes: bool = False,
    y_minutes_cumulative: bool = False,
) -> Figure | Axes:
    """
    DataFrame columns:
        pipeline_ref
        id: supervisor leaf stage id
        sample_time_year: sample year when this cost was recorded
        duration: cost of the pipeline at that time
    """
    sns.set_theme(style="whitegrid")
    init_plot()
    setup_font(small_label=True, small_title=True, small_ticks=True)

    fig, axs = plt.subplots(
        nrows=len(pipeline_ids),
        ncols=2,
        edgecolor="black",
        frameon=True,
        figsize=(
            DOUBLE_FIG_WIDTH * width_factor,
            2 * DOUBLE_FIG_HEIGHT * height_factor,
        ),
        dpi=600,
    )

    x_col = "sample_time_year"
    y_col = "duration"
    hue_col = "id"

    palette = sns.color_palette("RdBu", 10)
    new_palette = {
        "train": palette[0],
        "inform remaining data": palette[-2],
        "evaluate trigger policy": palette[2],
        "inform trigger": palette[-1],
        "store trained model": palette[1],
    }

    # use sum of all pipelines to determine the order of the bars that is consistent across subplots
    df_agg = df_costs.groupby([hue_col]).agg({y_col: "sum"}).reset_index()
    df_agg = df_agg.sort_values(y_col, ascending=False)
    categories = df_agg[hue_col].unique()

    legend_tuple = (pipeline_ids[0], True)

    for row, pipeline_id in enumerate(pipeline_ids):
        # sort by cumulative duration
        df_costs_pipeline = df_costs[df_costs["pipeline_ref"] == f"pipeline_{pipeline_id}"]

        for cumulative in [False, True]:
            df_final = df_costs_pipeline.copy()
            if cumulative and y_minutes_cumulative:
                df_final[y_col] = df_final[y_col] / 60
            elif not cumulative and y_minutes:
                df_final[y_col] = df_final[y_col] / 60

            ax = axs[row, int(cumulative)] if len(pipeline_ids) > 1 else axs[int(cumulative)]
            h = sns.histplot(
                df_final,
                x=x_col,
                weights=y_col,
                bins=2014 - 1930 + 1,
                cumulative=cumulative,
                # discrete=True,
                multiple="stack",
                linewidth=0,  # Remove white edges between bars
                shrink=1.0,  # Ensure bars touch each other
                alpha=1.0,  # remove transparaency
                # hue
                hue="id",
                hue_order=categories,
                palette=new_palette,
                # ax=axs[int(cumulative)],  # for 1 pipeline, only 1 row
                ax=ax,
                # legend
                legend=legend_tuple == (pipeline_id, cumulative),
                zorder=-2,
            )

            # Rasterize the heatmap background to avoid anti-aliasing artifacts
            for bar in h.patches:
                bar.set_rasterized(True)

            h.grid(axis="y", linestyle="--", alpha=grid_alpha, zorder=3, color="lightgray")
            h.grid(axis="x", linestyle="--", alpha=grid_alpha, zorder=3, color="lightgray")

            if len(title_map) > 0:
                # size huge
                h.set_title(title_map[pipeline_id])

            # # Set x-axis
            h.set(xlim=x_lim)
            h.set_xlabel(x_label, labelpad=10)

            if x_date_locator:
                h.xaxis.set_major_locator(x_date_locator)
                # ax.set_xticklabels(x_ticks, rotation=0)
                h.xaxis.set_major_formatter(x_date_formatter)
                # ticks = ax.get_xticks()
                plt.xticks(rotation=0)
            elif x_ticks is not None:
                h.set_xticks(
                    ticks=x_ticks,
                    labels=x_ticks,
                    rotation=0,
                    # ha='right'
                )

            if cumulative:
                h.set_ylabel(cumulative_ylabel, labelpad=20)
                if y_lim_cumulative:
                    h.set(ylim=y_lim_cumulative)
                if y_ticks_cumulative:
                    h.set_yticks(ticks=y_ticks_cumulative, labels=y_ticks_cumulative, rotation=0)
                else:
                    h.yaxis.set_major_locator(MaxNLocator(nbins=4))
            else:
                h.set_ylabel(duration_ylabel, labelpad=20)
                if y_ticks:
                    h.set_yticks(ticks=y_ticks, labels=y_ticks, rotation=0)
                else:
                    h.yaxis.set_major_locator(MaxNLocator(nbins=4))
            if legend_tuple == (pipeline_id, cumulative):
                # set hue label
                legend = h.get_legend()

                legend.set_title("")  # remove title

                # expand legend horizontally
                # legend.set_bbox_to_anchor((0, 1, 1, 0), transform=h.transAxes)

    # Display the plot
    plt.tight_layout()

    return fig
