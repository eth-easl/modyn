import dataclasses
from dataclasses import dataclass

import pandas as pd
import plotly.express as px
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


_shared_data = _SharedData()

# -------------------------------------------------------------------------------------------------------------------- #
#                                                        FIGURE                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


def gen_figure(
    page: str, multi_pipeline_mode: bool, patch_yearbook: bool, eval_handler: str, dataset_id: str, metric: str
) -> go.Figure:
    """
    Create the evaluation over time figure with a line plot.

    Args:
        page: Page name where the plot is displayed
        multi_pipeline_mode: only displays the measurements of the active model of a pipeline at every time;
            the color dimension will be the pipeline id and not the model id
        patch_yearbook: convert the time axis from days
        eval_handler: Evaluation handler reference
        dataset_id: Dataset id
        metric: Evaluation metric (replaced with facet)
    """
    df_adjusted = _shared_data.df_logs_eval_single[page].copy()
    df_adjusted = df_adjusted[
        (df_adjusted["dataset_id"] == dataset_id)
        & (df_adjusted["eval_handler"] == eval_handler)
        # & (df_adjusted["metric"] == metric)
    ]

    # Yearbook as a mapped time dimension (to display the correct timestamps we need to convert back from days to years)
    if patch_yearbook:
        for column in ["interval_center", "interval_start", "interval_end"]:
            patch_yearbook_time(df_adjusted, column)

    if multi_pipeline_mode:
        # we only want the pipeline performance (composed of the models active periods stitched together)
        df_adjusted = df_adjusted[df_adjusted["currently_active_model"]]
    else:
        assert df_adjusted["pipeline_ref"].nunique() == 1
        # add the pipeline time series which is the performance of different models stitched together dep.
        # w.r.t which model was active
        pipeline_composite_model = df_adjusted[df_adjusted["currently_active_model"]]
        pipeline_composite_model["model_idx"] = "00-pipeline-composite-model"
        number_digits = len(str(df_adjusted["model_idx"].max()))
        df_adjusted["model_idx"] = df_adjusted["model_idx"].astype(str).str.zfill(number_digits)
        df_adjusted = pd.concat([df_adjusted, pipeline_composite_model])

    fig = px.line(
        df_adjusted,
        x="interval_center",
        y="value",
        color="pipeline_ref" if multi_pipeline_mode else "model_idx",
        facet_row="metric",
        markers=True,
        labels={
            "interval_center": "Time (samples)",
            "metric": "Evaluation metric",
            "value": "Evaluation metric value",
            "pipeline_ref": "Pipeline",
            "model_idx": "Model",
        },
        category_orders={
            "pipeline_ref": list(sorted(df_adjusted["pipeline_ref"].unique())),
            "model_idx": list(sorted(df_adjusted["model_idx"].unique())),
        },
        hover_data=df_adjusted.columns,
        height=400 * len(df_adjusted["metric"].unique()),
    )

    fig.update_yaxes(matches=None, showticklabels=True)  # y axis should be independent (different metrics)
    fig.update_xaxes(showticklabels=True, nticks=40)
    return fig


# -------------------------------------------------------------------------------------------------------------------- #
#                                                      UI SECTION                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


def section_metricovertime(page: str, multi_pipeline_mode: bool, df_logs_eval_single: pd.DataFrame) -> html.Div:
    _shared_data.df_logs_eval_single[page] = df_logs_eval_single

    @callback(
        Output(f"{page}-evalovertime-plot", "figure"),
        Input(f"{page}-evalovertime-radio-time-patch-yearbook", "value"),
        Input(f"{page}-evalovertime-evaluation-handler", "value"),
        Input(f"{page}-evalovertime-dataset-id", "value"),
        Input(f"{page}-evalovertime-evaluation-metric", "value"),
    )
    def update_figure(patch_yearbook: bool, eval_handler_ref: str, dataset_id: str, metric: str) -> go.Figure:
        return gen_figure(page, multi_pipeline_mode, patch_yearbook, eval_handler_ref, dataset_id, metric)

    eval_handler_refs = list(df_logs_eval_single["eval_handler"].unique())
    eval_datasets = list(df_logs_eval_single["dataset_id"].unique())
    eval_metrics = list(df_logs_eval_single["metric"].unique())

    return html.Div(
        [
            dcc.Markdown("## Evaluation metrics over time"),
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
                                id=f"{page}-evalovertime-evaluation-handler",
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
                                id=f"{page}-evalovertime-dataset-id",
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
                                id=f"{page}-evalovertime-evaluation-metric",
                                options=[
                                    {"label": metric, "value": metric, "disabled": True} for metric in eval_metrics
                                ],
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
                                id=f"{page}-evalovertime-radio-time-patch-yearbook",
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
                id=f"{page}-evalovertime-plot",
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
