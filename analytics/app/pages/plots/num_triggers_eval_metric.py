import dataclasses

import pandas as pd
import plotly.express as px
from analytics.app.data.transform import df_aggregate_eval_metric
from dash import Input, Output, callback, dcc, html
from modyn.supervisor.internal.grpc.enums import PipelineStage
from plotly import graph_objects as go


@dataclasses.dataclass
class _SharedData:
    """We use the call by reference features asa the callbacks in the UI are not updated over the lifetime of the app.
    Therefore the need a reference to the data structure at startup time (even though data is not available yet).
    """

    df_logs_agg: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    df_logs_eval_single: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    """page, data"""


_shared_data = _SharedData()

# -------------------------------------------------------------------------------------------------------------------- #
#                                                        FIGURE                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


def gen_fig_scatter_num_triggers(
    page: str,
    multi_pipeline_mode: bool,
    eval_handler: str,
    dataset_id: str,
    metric: str,
    aggregate_metric: bool = True,
    time_weighted: bool = True,
    only_active_periods: bool = True,
) -> go.Figure:
    """
    Args:
        page: Page name where the plot is displayed
        multi_pipeline_mode: only displays the measurements of the active model of a pipeline at every time;
            the color dimension will be the pipeline id and not the model id
        eval_handler: Evaluation handler reference
        dataset_id: Dataset id
        metric: Evaluation metric (replaced with facet)
        aggregate_metric: Whether to aggregate the metric over all evaluations (scatter plot instead of boxplot)
        time_weighted: Whether to weight the aggregation by the evaluation interval length
    """
    # unpack data
    df_logs_agg = _shared_data.df_logs_agg[page]

    df_logs_eval_single = _shared_data.df_logs_eval_single[page]
    df_logs_eval_single = df_logs_eval_single[
        (df_logs_eval_single["dataset_id"] == dataset_id)
        & (df_logs_eval_single["eval_handler"] == eval_handler)
        # & (df_adjusted["metric"] == metric)
    ]

    if multi_pipeline_mode or only_active_periods:
        # we only want the pipeline performance (composed of the models active periods stitched together)
        df_logs_eval_single = df_logs_eval_single[df_logs_eval_single["currently_active_model"]]

    if not multi_pipeline_mode:
        assert df_logs_eval_single["pipeline_ref"].nunique() == 1

        # add the pipeline time series which is the performance of different models stitched together dep.
        # w.r.t which model was active
        pipeline_composite_model = df_logs_eval_single[df_logs_eval_single["currently_active_model"]]
        pipeline_composite_model["id_model"] = "0-pipeline-composite-model"
        df_logs_eval_single["id_model"] = df_logs_eval_single["id_model"].astype(str)
        df_logs_eval_single = pd.concat([df_logs_eval_single, pipeline_composite_model])

    col_map = {"value": "metric_value", "count": "num_triggers"}
    num_triggers = df_logs_agg[df_logs_agg["id"] == PipelineStage.HANDLE_SINGLE_TRIGGER.name][["pipeline_ref", "count"]]
    accuracies = df_logs_eval_single
    labels = {
        "pipeline_ref": "Pipeline",
        "metric": "Metric",
        "num_triggers": "#triggers (proxy for cost)",
        "metric_value": f"Metric value {'(mean)' if aggregate_metric else ''}",
    }
    category_orders = {
        "pipeline_ref": list(sorted(accuracies["pipeline_ref"].unique())),
        "id_model": list(sorted(accuracies["id_model"].unique())),
    }
    if aggregate_metric:
        mean_accuracies = df_aggregate_eval_metric(
            accuracies,
            group_by=["pipeline_ref", "metric"] + (["id_model"] if not multi_pipeline_mode else []),
            in_col="value",
            out_col="metric_value",
            aggregate_func="time_weighted_avg" if time_weighted else "mean",
        )
        merged = num_triggers.merge(mean_accuracies, on="pipeline_ref").rename(columns=col_map, inplace=False)
        fig = px.scatter(
            merged,
            x="num_triggers",
            y="metric_value",
            color="pipeline_ref" if multi_pipeline_mode else "id_model",
            facet_col="metric",
            labels=labels,
            category_orders=category_orders,
        )
    else:
        merged = num_triggers.merge(accuracies, on="pipeline_ref").rename(columns=col_map, inplace=False)
        fig = px.box(
            merged,
            x="num_triggers",
            y="metric_value",
            color="pipeline_ref" if multi_pipeline_mode else "id_model",
            facet_col="metric",
            labels=labels,
            category_orders=category_orders,
        )

    fig.update_yaxes(matches=None, showticklabels=True)  # y axis should be independent (different metrics)
    return fig


# -------------------------------------------------------------------------------------------------------------------- #
#                                                      UI SECTION                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


def section3_scatter_num_triggers(
    page: str, multi_pipeline_mode: bool, df_logs_agg: pd.DataFrame, df_logs_eval_single: pd.DataFrame
) -> html.Div:
    assert "pipeline_ref" in df_logs_agg.columns.tolist()
    assert "pipeline_ref" in df_logs_eval_single.columns.tolist()
    _shared_data.df_logs_agg[page] = df_logs_agg
    _shared_data.df_logs_eval_single[page] = df_logs_eval_single

    @callback(
        Output(f"{page}-scatter-plot-num-triggers", "figure"),
        Input(f"{page}-radio-scatter-number-triggers-evaluation-handler", "value"),
        Input(f"{page}-radio-scatter-number-triggers-dataset-id", "value"),
        Input(f"{page}-radio-scatter-number-triggers-metric", "value"),
        Input(f"{page}-radio-scatter-number-triggers-agg-y", "value"),
        Input(f"{page}-radio-1d-eval-metric-only-active-model-periods", "value"),
        Input(f"{page}-radio-scatter-number-triggers-only-active-model-periods", "value"),
    )
    def update_scatter_num_triggers(
        eval_handler_ref: str,
        dataset_id: str,
        metric: str,
        aggregate_metric: bool,
        time_weighted: bool,
        only_active_periods: bool = True,
    ) -> go.Figure:
        return gen_fig_scatter_num_triggers(
            page,
            multi_pipeline_mode,
            eval_handler_ref,
            dataset_id,
            metric,
            aggregate_metric,
            time_weighted,
            only_active_periods,
        )

    eval_handler_refs = list(df_logs_eval_single["eval_handler"].unique())
    eval_datasets = list(df_logs_eval_single["dataset_id"].unique())
    eval_metrics = list(df_logs_eval_single["metric"].unique())

    return html.Div(
        [
            dcc.Markdown(
                """
        ### Number of triggers vs. evaluation metric (e.g. accuracy)

        _Select the metric to plot against the number of triggers._
    """
            ),
            (
                html.Div(
                    [
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
                                            id=f"{page}-radio-scatter-number-triggers-evaluation-handler",
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
                                            id=f"{page}-radio-scatter-number-triggers-dataset-id",
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
                                                #### Evaluation metric (y-axis)
                                            """
                                        ),
                                        dcc.RadioItems(
                                            id=f"{page}-radio-scatter-number-triggers-metric",
                                            options=[
                                                {"label": metric, "value": metric, "disabled": True}
                                                for metric in eval_metrics
                                            ],
                                            value=eval_metrics[0],
                                            persistence=True,
                                        ),
                                    ],
                                    style={"padding-right": "50px"},
                                ),
                                html.Td(
                                    [
                                        dcc.Markdown(
                                            """
                                                #### Agg. metric (y-axis)
                                                over all evaluations
                                            """
                                        ),
                                        dcc.RadioItems(
                                            id=f"{page}-radio-scatter-number-triggers-agg-y",
                                            options=[
                                                {"label": "yes (mean)", "value": True},
                                                {"label": "no (boxplot)", "value": False},
                                            ],
                                            value=True,
                                            persistence=True,
                                        ),
                                    ],
                                    style={"padding-right": "50px"},
                                ),
                                html.Td(
                                    [
                                        dcc.Markdown(
                                            """
                                                #### aggregation weights
                                                Only applies when aggregating the metric
                                            """
                                        ),
                                        dcc.RadioItems(
                                            id=f"{page}-radio-scatter-number-triggers-agg-time-weighted",
                                            options=[
                                                {"label": "eval interval length (time weighted mean)", "value": True},
                                                {"label": "equal weights (simple mean)", "value": False},
                                            ],
                                            value=True,
                                            persistence=True,
                                        ),
                                    ],
                                    style={"padding-right": "50px"},
                                ),
                                html.Td(
                                    [
                                        dcc.Markdown(
                                            """
                                                #### Only active model periods
                                                Aggregate only evaluation result in the period where the model was active.
                                            """
                                        ),
                                        dcc.RadioItems(
                                            id=f"{page}-radio-scatter-number-triggers-only-active-model-periods",
                                            options=[{"label": "Yes", "value": True}, {"label": "No", "value": False}],
                                            value=True,
                                            persistence=True,
                                        ),
                                    ],
                                    style={"padding-right": "50px"},
                                    hidden=multi_pipeline_mode,
                                ),
                            ]
                        ),
                        dcc.Graph(
                            id=f"{page}-scatter-plot-num-triggers",
                            figure=gen_fig_scatter_num_triggers(
                                page,
                                multi_pipeline_mode,
                                eval_handler=eval_handler_refs[0] if len(eval_handler_refs) > 0 else None,
                                dataset_id=eval_datasets[0] if len(eval_datasets) > 0 else None,
                                metric=eval_metrics[0] if len(eval_metrics) > 0 else None,
                                aggregate_metric=False,
                            ),
                        ),
                    ]
                )
                if eval_metrics
                else dcc.Markdown("No evaluation metrics found.")
            ),
        ]
    )
