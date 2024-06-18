import dataclasses
from dataclasses import dataclass

import pandas as pd
from analytics.app.data.transform import patch_yearbook_time
from dash import Input, Output, callback, dcc, html
from plotly import graph_objects as go


@dataclass
class _SharedData:
    """We use the call by reference features asa the callbacks in the UI are not updated over the lifetime of the app.
    Therefore the need a reference to the data structure at startup time (even though data is not available yet).
    """

    df_logs_eval_single: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    """page, data"""

    df_logs_models: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    """page, data"""


_shared_data = _SharedData()

# -------------------------------------------------------------------------------------------------------------------- #
#                                                        FIGURE                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


def gen_figure(
    page: str, multi_pipeline_mode: bool, patch_yearbook: bool, eval_handler: str, dataset_id: str, metric: str
) -> go.Figure:
    """
    Create the cost over time figure with barplot or histogram. Histogram has nice binning while barplot is precise.

    Args:
        page: Page name where the plot is displayed
        multi_pipeline_mode: only displays the measurements of the active model of a pipeline at every time;
            the color dimension will be the pipeline id and not the model id
        patch_yearbook: convert the time axis from days
        eval_handler: Evaluation handler reference
        dataset_id: Dataset id
        metric: Evaluation metric (replaced with facet)
    """
    df_logs_models = _shared_data.df_logs_models[page].copy()
    df_adjusted = _shared_data.df_logs_eval_single[page].copy()
    df_adjusted = df_adjusted[
        (df_adjusted["dataset_id"] == dataset_id)
        & (df_adjusted["eval_handler"] == eval_handler)
        & (df_adjusted["metric"] == metric)
    ]

    # Yearbook as a mapped time dimension (to display the correct timestamps we need to convert back from days to years)
    if patch_yearbook:
        for column in ["interval_center", "interval_start", "interval_end", "sample_time", "sample_time_until"]:
            patch_yearbook_time(df_adjusted, column)

    df_adjusted = df_adjusted.sort_values(by=["interval_center"])

    if multi_pipeline_mode:
        # we only want the pipeline performance (composed of the models active periods stitched together)
        df_adjusted = df_adjusted[df_adjusted["most_recent_model"]]

        # in model dataframe convert pipeline_ref to pipeline_id as we need int for the heatmap
        df_adjusted["pipeline_id"] = df_adjusted["pipeline_ref"].str.split("-").str[0].astype(int)
        df_logs_models["pipeline_id"] = df_logs_models["pipeline_ref"].str.split("-").str[0].astype(int)

    else:
        assert df_adjusted["pipeline_ref"].nunique() == 1
        # add the pipeline time series which is the performance of different models stitched together dep.
        # w.r.t which model was active
        pipeline_composite_model = df_adjusted[df_adjusted["most_recent_model"]]
        pipeline_composite_model["model_idx"] = "0-pipeline-composite-model"

    # build heatmap matrix dataframe:
    heatmap_data = df_adjusted.pivot(
        index=["model_idx"] if not multi_pipeline_mode else ["pipeline_id"], columns="interval_center", values="value"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale="RdBu_r",
        )
    )
    fig.update_layout(
        xaxis_nticks=40,
        yaxis_nticks=2 * min(20, len(heatmap_data.index)),
        width=2200,
        height=1100,
        # "pipeline_id": "Pipeline",
        # "metric": "Metric",
        # "interval_center": "Evaluation time (interval center)",
    )
    shapes = []

    if not multi_pipeline_mode:
        # diagonal 1
        shapes += [
            dict(
                type="line",
                x0=active_[1]["interval_start"],
                y0=active_[1]["model_idx"] - 0.5,
                x1=active_[1]["interval_end"],
                y1=active_[1]["model_idx"] + 0.5,
                line=dict(color="Green", width=5),
            )
            for active_ in df_adjusted[
                df_adjusted["most_recent_model"]
            ].iterrows()  # if "pipeline-composite-model" not in active_[1]["id_model"]
        ]
        # diagonal 2
        shapes += [
            dict(
                type="line",
                x0=active_[1]["interval_start"],
                y0=active_[1]["model_idx"] + 0.5,
                x1=active_[1]["interval_end"],
                y1=active_[1]["model_idx"] - 0.5,
                line=dict(color="Green", width=5),
            )
            for active_ in df_adjusted[
                df_adjusted["most_recent_model"]
            ].iterrows()  # if "pipeline-composite-model" not in active_[1]["id_model"]
        ]

    # Model training periods
    y_column = "pipeline_id" if multi_pipeline_mode else "model_idx"
    for type_ in ["usage"] if multi_pipeline_mode else ["train", "usage"]:
        shapes += [
            dict(
                type="rect",
                x0=active_[1][f"{type_}_start"],
                x1=active_[1][f"{type_}_end"],
                y0=active_[1][y_column] - 0.5,
                y1=active_[1][y_column] + 0.5,
                line=dict(color="Orange" if type_ == "train" else "Black", width=4),
            )
            for active_ in df_logs_models.iterrows()
        ]
    fig.update_layout(shapes=shapes)
    return fig


# -------------------------------------------------------------------------------------------------------------------- #
#                                                      UI SECTION                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


def section_evalheatmap(
    page: str, multi_pipeline_mode: bool, df_logs_eval_single: pd.DataFrame, df_logs_models: pd.DataFrame
) -> html.Div:
    _shared_data.df_logs_eval_single[page] = df_logs_eval_single
    _shared_data.df_logs_models[page] = df_logs_models

    @callback(
        Output(f"{page}-evalheatmap-plot", "figure"),
        Input(f"{page}-evalheatmap-radio-time-patch-yearbook", "value"),
        Input(f"{page}-evalheatmap-evaluation-handler", "value"),
        Input(f"{page}-evalheatmap-dataset-id", "value"),
        Input(f"{page}-evalheatmap-evaluation-metric", "value"),
    )
    def update_figure(patch_yearbook: bool, eval_handler_ref: str, dataset_id: str, metric: str) -> go.Figure:
        return gen_figure(page, multi_pipeline_mode, patch_yearbook, eval_handler_ref, dataset_id, metric)

    eval_handler_refs = list(df_logs_eval_single["eval_handler"].unique())
    eval_datasets = list(df_logs_eval_single["dataset_id"].unique())
    eval_metrics = list(df_logs_eval_single["metric"].unique())

    return html.Div(
        [
            dcc.Markdown(
                """
                    ## Evaluation heatmap
                    evaluations over all models and the evaluation samples

                    Markers:
                    - reen cross: Evaluation request / interval with a currently active model (used for inference)
                    - Orange box: Interval of training data for a particular model (independent from evaluation)
                    - Black box: Model usage interval (used for inference; independent from evaluation)

                    In the pipeline comparison mode we only show the black boxes so one can see when models are switched
                """
            ),
            html.Tr(
                [
                    html.Td(
                        [
                            dcc.Markdown(
                                """
                                    #### Evaluation handler
                                """
                            ),
                            dcc.RadioItems(
                                id=f"{page}-evalheatmap-evaluation-handler",
                                options=eval_handler_refs,
                                value=eval_handler_refs[0] if len(eval_handler_refs) > 0 else None,
                                persistence=True,
                            ),
                        ],
                        style={"padding-right": "50px"},
                    ),
                    html.Td(
                        [
                            dcc.Markdown(
                                """
                                    #### Dataset_id
                                """
                            ),
                            dcc.RadioItems(
                                id=f"{page}-evalheatmap-dataset-id",
                                options=eval_datasets,
                                value=eval_datasets[0] if len(eval_datasets) > 0 else None,
                                persistence=True,
                            ),
                        ],
                        style={"padding-right": "50px"},
                    ),
                    html.Td(
                        [
                            dcc.Markdown(
                                """
                                    #### Evaluation metric
                                """
                            ),
                            dcc.RadioItems(
                                id=f"{page}-evalheatmap-evaluation-metric",
                                options=eval_metrics,
                                value=eval_metrics[0] if len(eval_metrics) > 0 else None,
                                persistence=True,
                            ),
                        ],
                        style={"padding-right": "50px"},
                    ),
                    html.Td(
                        [
                            dcc.Markdown(
                                """
                                    #### Patch sample time (for yearbook)
                                """
                            ),
                            dcc.RadioItems(
                                id=f"{page}-evalheatmap-radio-time-patch-yearbook",
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
                ]
            ),
            dcc.Graph(
                id=f"{page}-evalheatmap-plot",
                figure=gen_figure(
                    page,
                    multi_pipeline_mode,
                    False,
                    eval_handler=eval_handler_refs[0] if len(eval_handler_refs) > 0 else None,
                    dataset_id=eval_datasets[0] if len(eval_datasets) > 0 else None,
                    metric=eval_metrics[0] if len(eval_metrics) > 0 else None,
                ),
            ),
        ]
    )
