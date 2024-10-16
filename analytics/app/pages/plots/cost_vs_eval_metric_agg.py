from dataclasses import dataclass
from typing import Any, get_args

import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html
from plotly import graph_objects as go

from analytics.app.data.const import CompositeModelOptions
from analytics.app.data.transform import AGGREGATION_FUNCTION, EVAL_AGGREGATION_FUNCTION, df_aggregate_eval_metric


@dataclass
class _PageState:
    """Callbacks cannot be updated after the initial rendering therefore we
    need to define and update state within global references."""

    df_all: pd.DataFrame
    df_agg_leaf: pd.DataFrame
    df_eval_single: pd.DataFrame

    composite_model_variant: CompositeModelOptions = "currently_active_model"


_shared_data: dict[str, _PageState] = {}  # page -> _PageState


# -------------------------------------------------------------------------------------------------------------------- #
#                                                        FIGURE                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


def gen_fig_scatter_num_triggers(
    page: str,
    eval_handler: str | Any | None,
    dataset_id: str | Any | None,
    metric: str | Any | None,
    agg_func_x: AGGREGATION_FUNCTION,
    agg_func_y: EVAL_AGGREGATION_FUNCTION,
    stages: list[str],
) -> go.Figure:
    # unpack data
    composite_model_variant = _shared_data[page].composite_model_variant
    df_all = _shared_data[page].df_all
    df_eval_single = _shared_data[page].df_eval_single
    df_eval_single = df_eval_single[
        (df_eval_single["dataset_id"] == dataset_id)
        & (df_eval_single["eval_handler"] == eval_handler)
        & (df_eval_single[composite_model_variant])
        # & (df_adjusted["metric"] == metric)
    ]

    agg_eval_metric = df_aggregate_eval_metric(
        df_eval_single,
        group_by=["pipeline_ref", "metric"],
        in_col="value",
        out_col="metric_value",
        aggregate_func=agg_func_y,
    )

    agg_duration = (
        df_all[df_all["id"].isin(stages)].groupby(["pipeline_ref"]).agg(cost=("duration", agg_func_x)).reset_index()
    )

    merged = agg_eval_metric.merge(agg_duration, on="pipeline_ref")
    assert (
        agg_eval_metric.shape[0] == merged.shape[0] == agg_duration.shape[0] * len(agg_eval_metric["metric"].unique())
    )
    fig = px.scatter(
        merged,
        x="cost",
        y="metric_value",
        color="pipeline_ref",
        facet_col="metric",
        labels={
            "cost": f"{agg_func_x} duration in sec. (proxy for cost)",
            "metric_value": f"{agg_func_y}",
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
    page: str,
    df_all: pd.DataFrame,
    df_agg_leaf: pd.DataFrame,
    df_eval_single: pd.DataFrame,
    composite_model_variant: CompositeModelOptions,
) -> html.Div:
    assert "pipeline_ref" in list(df_all.columns)
    assert "pipeline_ref" in list(df_eval_single.columns)

    if page not in _shared_data:
        _shared_data[page] = _PageState(
            composite_model_variant=composite_model_variant,
            df_all=df_all,
            df_agg_leaf=df_agg_leaf,
            df_eval_single=df_eval_single,
        )
    _shared_data[page].composite_model_variant = composite_model_variant
    _shared_data[page].df_all = df_all
    _shared_data[page].df_agg_leaf = df_agg_leaf
    _shared_data[page].df_eval_single = df_eval_single

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

    eval_handler_refs = list(df_eval_single["eval_handler"].unique())
    eval_datasets = list(df_eval_single["dataset_id"].unique())
    eval_metrics = list(df_eval_single["metric"].unique())

    stages = list(df_agg_leaf["id"].unique())

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
