from dataclasses import dataclass

import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html
from plotly import graph_objects as go

from analytics.app.data.transform import patch_yearbook_time


@dataclass
class _PageState:
    """Callbacks cannot be updated after the initial rendering therefore we
    need to define and update state within global references."""

    df_leaf: pd.DataFrame


_shared_data: dict[str, _PageState] = {}  # page -> _PageState


# -------------------------------------------------------------------------------------------------------------------- #
#                                                        FIGURE                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


def gen_figure(
    page: str,
    time_metric: str,
    cumulative: bool,
    histogram: bool = False,
    nbins: int = 65,
    patch_yearbook: bool = False,
) -> go.Figure:
    """Create the cost over time figure with barplot or histogram. Histogram
    has nice binning while barplot is precise.

    Args:
        page: Page name where the plot is displayed
        time_metric: Metric to use for the x axis chosen by the user
        cumulative: Whether the user selected a cumulative plot
        histogram: Whether to use histogram over barplot
        nbins: Number of bins; only used in the histogram=True case
    """
    df_adjusted = _shared_data[page].df_leaf.copy()  # TODO: remove
    if cumulative and not histogram:
        # as bar plots don't support cumulation natively

        # first, we want to ensure every group ('pipeline_ref', 'id') has exactly one value per time step
        # we impute missing values with 0 (no cost occurred)
        all_times = df_adjusted[[time_metric]].drop_duplicates(inplace=False).reset_index(drop=True)
        all_groups = df_adjusted[["pipeline_ref", "id"]].drop_duplicates(inplace=False).reset_index(drop=True)

        # build cartesian product for index
        all_combinations = all_groups.merge(all_times, how="cross")
        df_adjusted = all_combinations.merge(df_adjusted, on=["pipeline_ref", "id", time_metric], how="left").fillna(0)

        # now build cumulative sum
        df_adjusted = (
            df_adjusted.groupby(["pipeline_ref", "id", time_metric]).agg(duration=("duration", "sum")).reset_index()
        )
        df_adjusted = df_adjusted.sort_values(["pipeline_ref", "id", time_metric])
        df_adjusted["duration"] = df_adjusted.groupby(["pipeline_ref", "id"])["duration"].cumsum()

    # coloring in order of decreasing avg. duration
    avg_duration_per_stage = df_adjusted.groupby(["pipeline_ref", "id"])["duration"].mean().sort_values(ascending=False)
    df_adjusted = df_adjusted.merge(
        avg_duration_per_stage, on=["pipeline_ref", "id"], suffixes=("", "_avg")
    ).sort_values("duration_avg", ascending=False)

    # Yearbook as a mapped time dimension (to display the correct timestamps we need to convert back from days to years)
    if time_metric == "sample_time" and patch_yearbook:
        patch_yearbook_time(df_adjusted, "sample_time")

    labels = {
        "duration": "Duration (sek)" if not cumulative else "Cumulative duration (sek)",
        "accuracy": "Accuracy",
        "pipeline_ref": "Pipeline",
        "batch_idx": "Batch",
        "id": "Pipeline stage",
    }

    if histogram:
        fig = px.histogram(
            df_adjusted,
            x=time_metric,
            y="duration",
            color="id",
            facet_row="pipeline_ref",
            labels=labels,
            nbins=nbins,
            cumulative=cumulative,
            height=350 * len(df_adjusted["pipeline_ref"].unique()),
        )
    else:
        fig = px.bar(
            df_adjusted,
            x=time_metric,
            y="duration",
            color="id",
            facet_row="pipeline_ref",
            labels=labels,
            height=400 * len(df_adjusted["pipeline_ref"].unique()),
        )

    fig.update_layout(xaxis_nticks=40, yaxis_nticks=15)
    return fig


# -------------------------------------------------------------------------------------------------------------------- #
#                                                      UI SECTION                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


def section_cost_over_time(page: str, df_leaf: pd.DataFrame) -> html.Div:
    if page not in _shared_data:
        _shared_data[page] = _PageState(df_leaf=df_leaf)
    _shared_data[page].df_leaf = df_leaf

    @callback(
        Output(f"{page}-costovertime-plot", "figure"),
        Input(f"{page}-costovertime-radio-time-dimension", "value"),
        Input(f"{page}-costovertime-checkbox-cumulative", "value"),
        Input(f"{page}-costovertime-checkbox-histogram", "value"),
        Input(f"{page}-costovertime-nbins-slider", "value"),
        Input(f"{page}-costovertime-radio-time-patch-yearbook", "value"),
    )
    def update_figure(
        time_metric: str, cumulative: bool, histogram: bool, nbins: int, patch_yearbook: bool
    ) -> go.Figure:
        return gen_figure(page, time_metric, cumulative, histogram, nbins, patch_yearbook)

    @callback(
        Output(f"{page}-costovertime-nbins-slider", "disabled"),
        Input(f"{page}-costovertime-checkbox-histogram", "value"),
    )
    def hide_bin_slider(histogram: bool) -> bool:
        return not histogram

    time_metrics = {
        "batch_idx": "Batch index",
        "sample_time": "Sample time",
        "sample_idx": "Sample index",
        "trigger_idx": "Trigger index",
    }

    return html.Div(
        [
            dcc.Markdown(
                """
            ## Cost-/Accuracy tradeoff over time

            Using a time-based x-axis we show how the cost and accuracy of a pipeline run changes over a pipeline run.

            We currencyly support three time dimensions:
            - Realtime (datetime): The actual time the pipeline run was executed
            - Sample Time (index or unix timestamp) The time the pipeline run was scheduled to be executed
            - Batch Index (int): The batch index of the pipeline run
            - Trigger index: group costs by trigger index (we can e.g. see that if not triggering often we process many batches between to triggers)
        """
            ),
            html.Tr(
                [
                    html.Td(
                        [
                            dcc.Markdown(
                                """
                    #### Time dimension metric (x-axis)

                    _**Note**: some metrics like trigger index might not represent an even time distribution._
                """
                            ),
                            dcc.RadioItems(
                                id=f"{page}-costovertime-radio-time-dimension",
                                options=time_metrics,
                                value="batch_idx",
                                persistence=True,
                            ),
                            dcc.Markdown(
                                """
                    #### Patch sample time (for yearbook)
                """
                            ),
                            dcc.RadioItems(
                                id=f"{page}-costovertime-radio-time-patch-yearbook",
                                options=[
                                    {"label": "yes (convert day based timestamps to years)", "value": True},
                                    {"label": "no (use timestamps as they are)", "value": False},
                                ],
                                value=False,
                                persistence=True,
                            ),
                        ],
                        style={"padding-right": "50px"},
                    ),
                    html.Td(
                        [
                            dcc.Markdown(
                                """
                    #### Cumulative cost over time

                    _**Note**: The cumulative cost is the sum of all costs up to the current time._
                """
                            ),
                            dcc.RadioItems(
                                id=f"{page}-costovertime-checkbox-cumulative",
                                options=[
                                    {"label": "yes (cumulative)", "value": True},
                                    {"label": "no (single)", "value": False},
                                ],
                                value=False,
                                persistence=True,
                            ),
                        ],
                        style={"padding-right": "50px"},
                    ),
                    html.Td(
                        [
                            dcc.Markdown(
                                """
                    #### Boxplot or Histogram
                    Boxplot: has precise bars (no aggregation into bins)
                    Histogram: more pleasant visuals and dynamic binning
                """
                            ),
                            dcc.RadioItems(
                                id=f"{page}-costovertime-checkbox-histogram",
                                options=[
                                    {"label": "Histogram", "value": True},
                                    {"label": "Boxplot", "value": False},
                                ],
                                value=False,
                                persistence=True,
                            ),
                            html.Br(),
                            dcc.Slider(
                                id=f"{page}-costovertime-nbins-slider",
                                min=1,
                                max=1000,
                                value=65,
                                disabled=True,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]
                    ),
                ]
            ),
            dcc.Graph(id=f"{page}-costovertime-plot", figure=gen_figure(page, "sample_time", True)),
        ]
    )
