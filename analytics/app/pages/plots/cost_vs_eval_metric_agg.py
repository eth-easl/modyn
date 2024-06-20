import dataclasses
from typing import get_args

import pandas as pd
import plotly.express as px
from analytics.app.data.transform import AGGREGATION_FUNCTION, EVAL_AGGREGATION_FUNCTION, df_aggregate_eval_metric
from dash import Input, Output, callback, dcc, html
from plotly import graph_objects as go


@dataclasses.dataclass
class _SharedData:
    """We use the call by reference features asa the callbacks in the UI are not updated over the lifetime of the app.
    Therefore the need a reference to the data structure at startup time (even though data is not available yet).
    """

    df_logs: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    df_logs_agg_leaf: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    df_logs_eval_single: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    """page, data"""


_shared_data = _SharedData()

# -------------------------------------------------------------------------------------------------------------------- #
#                                                        FIGURE                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


def gen_fig_scatter_num_triggers(
    page: str,
    eval_handler: str,
    dataset_id: str,
    metric: str,
    agg_func_x: AGGREGATION_FUNCTION,
    agg_func_y: EVAL_AGGREGATION_FUNCTION,
    stages: list[str],
) -> go.Figure:
    # unpack data
    df_logs = _shared_data.df_logs[page]
    df_logs_eval_single = _shared_data.df_logs_eval_single[page].copy()
    df_logs_eval_single = df_logs_eval_single[
        (df_logs_eval_single["dataset_id"] == dataset_id)
        & (df_logs_eval_single["eval_handler"] == eval_handler)
        & (df_logs_eval_single["currently_active_model"])
        # & (df_adjusted["metric"] == metric)
    ]

    agg_eval_metric = df_aggregate_eval_metric(
        df_logs_eval_single,
        group_by=["pipeline_ref", "metric"],
        in_col="value",
        out_col="metric_value",
        aggregate_func=agg_func_y,
    )

    agg_duration = (
        df_logs[df_logs["id"].isin(stages)].groupby(["pipeline_ref"]).agg(cost=("duration", agg_func_x)).reset_index()
    )

    merged = agg_eval_metric.merge(agg_duration, on="pipeline_ref")
    fig = px.scatter(
        merged,
        x="cost",
        y="metric_value",
        color="pipeline_ref",
        facet_col="metric",
        labels={
            "cost": f"{agg_func_x} duration in sec. (proxy for cost)",
            "metric_value": f"{agg_func_y} {metric}",
            "pipeline_ref": "Pipeline",
        },
        category_orders={
            "pipeline_ref": list(sorted(agg_eval_metric["pipeline_ref"].unique())),
        },
    )

    fig.update_yaxes(matches=None, showticklabels=True)  # y axis should be independent (different metrics)
    return fig


# -------------------------------------------------------------------------------------------------------------------- #
#                                                      UI SECTION                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


def section3_scatter_cost_eval_metric(
    page: str, df_logs: pd.DataFrame, df_logs_agg_leaf: pd.DataFrame, df_logs_eval_single: pd.DataFrame
) -> html.Div:
    assert "pipeline_ref" in df_logs.columns.tolist()
    assert "pipeline_ref" in df_logs_eval_single.columns.tolist()
    _shared_data.df_logs[page] = df_logs
    _shared_data.df_logs_agg_leaf[page] = df_logs_agg_leaf
    _shared_data.df_logs_eval_single[page] = df_logs_eval_single

    @callback(
        Output(f"{page}-scatter-cost-eval", "figure"),
        Input(f"{page}-radio-scatter-cost-evaluation-handler", "value"),
        Input(f"{page}-radio-scatter-cost-dataset-id", "value"),
        Input(f"{page}-radio-scatter-cost-eval-metric", "value"),
        Input(f"{page}-radio-scatter-cost-eval-aggfunc-x", "value"),
        Input(f"{page}-radio-scatter-cost-eval-aggfunc-y", "value"),
        Input(f"{page}-radio-scatter-cost-stages", "value"),
    )
    def update_scatter_num_triggers(
        eval_handler_ref: str,
        dataset_id: str,
        metric_y: str,
        agg_func_x: AGGREGATION_FUNCTION,
        agg_func_y: EVAL_AGGREGATION_FUNCTION,
        stages: list[str],
    ) -> go.Figure:
        return gen_fig_scatter_num_triggers(
            page, eval_handler_ref, dataset_id, metric_y, agg_func_x, agg_func_y, stages
        )

    eval_handler_refs = list(df_logs_eval_single["eval_handler"].unique())
    eval_datasets = list(df_logs_eval_single["dataset_id"].unique())
    eval_metrics = list(df_logs_eval_single["metric"].unique())

    stages = list(df_logs_agg_leaf["id"].unique())

    return html.Div(
        [
            dcc.Markdown(
                """
        ### Cost vs. evaluation metric (e.g. accuracy)

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
                                            id=f"{page}-radio-scatter-cost-evaluation-handler",
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
                                            id=f"{page}-radio-scatter-cost-dataset-id",
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
                                            id=f"{page}-radio-scatter-cost-eval-metric",
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
                    #### Agg. func (x-axis)
                """
                                        ),
                                        dcc.RadioItems(
                                            id=f"{page}-radio-scatter-cost-eval-aggfunc-x",
                                            options=get_args(AGGREGATION_FUNCTION),
                                            value="sum",
                                            persistence=True,
                                        ),
                                    ],
                                    style={"padding-right": "50px"},
                                ),
                                html.Td(
                                    [
                                        dcc.Markdown(
                                            """
                    #### Agg. func (y-axis)
                """
                                        ),
                                        dcc.RadioItems(
                                            id=f"{page}-radio-scatter-cost-eval-aggfunc-y",
                                            options=get_args(EVAL_AGGREGATION_FUNCTION),
                                            value="sum",
                                            persistence=True,
                                        ),
                                    ],
                                    style={"padding-right": "50px"},
                                ),
                                html.Td(
                                    [
                                        dcc.Markdown(
                                            """
                    #### Pipeline stages
                    select pipe stages to include in sum
                """
                                        ),
                                        dcc.Checklist(
                                            id=f"{page}-radio-scatter-cost-stages",
                                            options=stages,
                                            value=stages,
                                            persistence=True,
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        dcc.Graph(
                            id=f"{page}-scatter-cost-eval",
                            figure=gen_fig_scatter_num_triggers(
                                page,
                                eval_handler=eval_handler_refs[0] if len(eval_handler_refs) > 0 else None,
                                dataset_id=eval_datasets[0] if len(eval_datasets) > 0 else None,
                                metric=eval_metrics[0] if len(eval_metrics) > 0 else None,
                                agg_func_x="sum",
                                agg_func_y="sum",
                                stages=stages,
                            ),
                        ),
                    ]
                )
                if eval_metrics
                else dcc.Markdown("No evaluation metrics found.")
            ),
        ]
    )
