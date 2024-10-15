from dataclasses import dataclass
from typing import Literal

import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html
from plotly import graph_objects as go

from analytics.app.data.const import CompositeModelOptions
from analytics.app.data.transform import patch_yearbook_time


@dataclass
class _PageState:
    """Callbacks cannot be updated after the initial rendering therefore we
    need to define and update state within global references."""

    df_models: pd.DataFrame
    df_eval_requests: pd.DataFrame

    composite_model_variant: CompositeModelOptions = "currently_active_model"


_shared_data: dict[str, _PageState] = {}  # page -> _PageState


# -------------------------------------------------------------------------------------------------------------------- #
#                                                        FIGURE                                                        #
# -------------------------------------------------------------------------------------------------------------------- #

YAxis = Literal["eval_samples", "train_samples", "train_batches"]


def gen_figure(
    page: str,
    multi_pipeline_mode: bool,
    time_metric: str | None,
    y_axis: YAxis | None,
    use_scatter_size: bool | None,
    patch_yearbook: bool,
    dataset_id: str | None,
    eval_handler: str | None,
) -> go.Figure:
    """Create the cost over time figure with barplot or histogram. Histogram
    has nice binning while barplot is precise.

    Args:
        page: Page name where the plot is displayed
        multi_pipeline_mode: only displays the measurements of the active model of a pipeline at every time;
        time_metric: Metric to use for the x axis chosen by the user
        y_axis: Metric to use for the y axis chosen by the user
        use_scatter_size: If True, the size of the scatter points is proportional to the number of samples
        patch_yearbook: If True, the time metric is patched to be a yearbook
    """
    composite_model_variant = _shared_data[page].composite_model_variant

    if y_axis == "eval_samples":
        df_evals = _shared_data[page].df_eval_requests
        df_evals = df_evals[(df_evals["dataset_id"] == dataset_id) & (df_evals["eval_handler"] == eval_handler)]

        if multi_pipeline_mode:
            df_evals = df_evals[df_evals[composite_model_variant]]

        # Yearbook as a mapped time dimension (to display the correct timestamps we need to convert back from days to years)
        if time_metric == "sample_time" and patch_yearbook:
            patch_yearbook_time(df_evals, "sample_time")

        fig = px.scatter(
            df_evals,
            x=time_metric,
            y="num_samples",
            color="pipeline_ref" if multi_pipeline_mode else "model_idx",
            size="num_samples" if use_scatter_size else None,
            hover_name="pipeline_ref" if multi_pipeline_mode else "model_idx",
            hover_data=df_evals.columns,
        )

    else:
        assert y_axis != "eval_center"

        # y_axis = "train_*""
        df_trainings = _shared_data[page].df_models.copy()  # TODO: remove copy

        # Yearbook as a mapped time dimension (to display the correct timestamps we need to convert back from days to years)
        if time_metric == "sample_time" and patch_yearbook:
            patch_yearbook_time(df_trainings, "sample_time")

        y_col = "num_samples" if y_axis == "train_samples" else "num_batches"

        fig = px.scatter(
            df_trainings,
            x=time_metric,
            y=y_col,
            color="pipeline_ref" if multi_pipeline_mode else "model_idx",
            size=y_col if use_scatter_size else None,
            hover_name="pipeline_ref" if multi_pipeline_mode else "model_idx",
            hover_data=df_trainings.columns,
        )

    fig.update_layout(xaxis_nticks=40, yaxis_nticks=15)
    return fig


# -------------------------------------------------------------------------------------------------------------------- #
#                                                      UI SECTION                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


def section_num_samples(
    page: str,
    multi_pipeline_mode: bool,
    df_models: pd.DataFrame,
    df_eval_requests: pd.DataFrame,
    composite_model_variant: CompositeModelOptions,
) -> html.Div:
    if page not in _shared_data:
        _shared_data[page] = _PageState(
            composite_model_variant=composite_model_variant,
            df_models=df_models,
            df_eval_requests=df_eval_requests,
        )
    _shared_data[page].composite_model_variant = composite_model_variant
    _shared_data[page].df_models = df_models
    _shared_data[page].df_eval_requests = df_eval_requests

    @callback(
        Output(f"{page}-num-samples-plot", "figure"),
        Input(f"{page}-num-samples-radio-time-dimension", "value"),
        Input(f"{page}-num-samples-y-axis", "value"),
        Input(f"{page}-num-samples-use-scatter-size", "value"),
        Input(f"{page}-num-samples-radio-time-patch-yearbook", "value"),
        # Evaluation related configs
        Input(f"{page}-num-samples-dataset-id", "value"),
        Input(f"{page}-num-samples-evaluation-handler", "value"),
    )
    def update_figure(
        time_metric: str,
        y_axis: YAxis,
        use_scatter_size: bool,
        patch_yearbook: bool,
        dataset_id: str,
        eval_handler: str,
    ) -> go.Figure:
        return gen_figure(
            page,
            multi_pipeline_mode,
            time_metric,
            y_axis,
            use_scatter_size,
            patch_yearbook,
            dataset_id,
            eval_handler,
        )

    @callback(
        Output(f"{page}-num-samples-eval-configs", "hidden"),
        Input(f"{page}-num-samples-y-axis", "value"),
    )
    def show_eval_config(y_axis: YAxis) -> bool:
        return y_axis != "eval_samples"

    time_metrics = {
        "batch_idx": "Batch index",
        "sample_time": "Sample time",
        "sample_idx": "Sample index",
        "trigger_idx": "Trigger index",
        "interval_center": "Evaluation interval center (only for y=eval_samples)",
    }

    eval_handler_refs = list(df_eval_requests["eval_handler"].unique())
    eval_datasets = list(df_eval_requests["dataset_id"].unique())

    return html.Div(
        [
            dcc.Markdown("""## Distribution of samples in datasets"""),
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
                                id=f"{page}-num-samples-radio-time-dimension",
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
                                id=f"{page}-num-samples-radio-time-patch-yearbook",
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
                            dcc.Markdown("#### y-axis"),
                            dcc.RadioItems(
                                id=f"{page}-num-samples-y-axis",
                                options=[
                                    {"label": "Number of samples in trigger", "value": "train_samples"},
                                    {"label": "Number of batches in trigger", "value": "train_batches"},
                                    {"label": "Number of samples in evaluation", "value": "eval_samples"},
                                ],
                                value="train_samples",
                                persistence=True,
                            ),
                            dcc.Markdown("#### y-axis"),
                            dcc.RadioItems(
                                id=f"{page}-num-samples-use-scatter-size",
                                options=[
                                    {"label": "Encode number of samples in size", "value": True},
                                    {"label": "Don't encode number of samples in size", "value": False},
                                ],
                                value=True,
                                persistence=True,
                            ),
                        ],
                        style={"padding-right": "50px"},
                    ),
                    html.Td(
                        id=f"{page}-num-samples-eval-configs",
                        hidden=True,
                        children=[
                            dcc.Markdown(
                                """
                                    #### Evaluation handler
                                """
                            ),
                            dcc.RadioItems(
                                id=f"{page}-num-samples-evaluation-handler",
                                options=eval_handler_refs,
                                value=eval_handler_refs[0] if len(eval_handler_refs) > 0 else None,
                                persistence=True,
                            ),
                            dcc.Markdown(
                                """
                                    #### Dataset_id
                                """
                            ),
                            dcc.RadioItems(
                                id=f"{page}-num-samples-dataset-id",
                                options=eval_datasets,
                                value=eval_datasets[0] if len(eval_datasets) > 0 else None,
                                persistence=True,
                            ),
                        ],
                    ),
                ]
            ),
            dcc.Graph(
                id=f"{page}-num-samples-plot",
                figure=gen_figure(
                    page,
                    multi_pipeline_mode,
                    "sample_time",
                    "train_samples",
                    use_scatter_size=True,
                    patch_yearbook=False,
                    eval_handler=eval_handler_refs[0] if len(eval_handler_refs) > 0 else None,
                    dataset_id=eval_datasets[0] if len(eval_datasets) > 0 else None,
                ),
            ),
        ]
    )
