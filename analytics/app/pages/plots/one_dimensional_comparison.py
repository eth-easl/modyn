from dataclasses import dataclass

import pandas as pd
import plotly.express as px
from analytics.app.data.const import CompositeModelOptions
from analytics.app.data.transform import OPTIONAL_EVAL_AGGREGATION_FUNCTION, df_aggregate_eval_metric
from dash import Input, Output, callback, dcc, html
from modyn.supervisor.internal.grpc.enums import PipelineStage
from plotly import graph_objects as go
from typing_extensions import get_args


@dataclass
class _PageState:
    """Callbacks cannot be updated after the initial rendering therefore we need to define and update state within
    global references.
    """

    df_all: pd.DataFrame
    df_eval_single: pd.DataFrame

    composite_model_variant: CompositeModelOptions = "currently_active_model"


_shared_data: dict[str, _PageState] = {}  # page -> _PageState


# -------------------------------------------------------------------------------------------------------------------- #
#                                                        FIGURE                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


def gen_fig_1d_cost(page: str) -> go.Figure:
    return px.box(
        _shared_data[page].df_all,
        x="pipeline_ref",
        y="duration",
        color="id",
        labels={"pipeline_ref": "Pipeline", "duration": "duration in seconds", "id": "Pipeline Stage"},
        title="Stage costs",
    )


def gen_figs_1d_eval(
    page: str,
    multi_pipeline_mode: bool,
    eval_handler: str,
    dataset_id: str,
    agg_func_eval_metric: OPTIONAL_EVAL_AGGREGATION_FUNCTION,
    only_active_periods: bool = True,
) -> go.Figure:
    composite_model_variant = _shared_data[page].composite_model_variant

    df_logs = _shared_data[page].df_all
    df_logs_eval_single = _shared_data[page].df_eval_single
    df_logs_eval_single = df_logs_eval_single[
        (df_logs_eval_single["dataset_id"] == dataset_id) & (df_logs_eval_single["eval_handler"] == eval_handler)
    ]

    if multi_pipeline_mode or only_active_periods:
        # we only want the pipeline performance (composed of the models active periods stitched together)
        df_logs_eval_single = df_logs_eval_single[df_logs_eval_single[composite_model_variant]]

    if not multi_pipeline_mode:
        assert df_logs_eval_single["pipeline_ref"].nunique() == 1

        digits = len(str(df_logs_eval_single["id_model"].max()))
        # fill with leading spaces to have a consistent sorting
        df_logs_eval_single["id_model"] = df_logs_eval_single["id_model"].astype(str).str.zfill(digits)

        # add the pipeline time series which is the performance of different models stitched together dep.
        # w.r.t which model was active
        pipeline_composite_model = df_logs_eval_single[df_logs_eval_single[composite_model_variant]]
        pipeline_composite_model["id_model"] = "0-pipeline-composite-model"
        df_logs_eval_single["id_model"] = df_logs_eval_single["id_model"].astype(str)
        df_logs_eval_single = pd.concat([df_logs_eval_single, pipeline_composite_model])

    if agg_func_eval_metric != "none":
        metrics = df_aggregate_eval_metric(
            df_logs_eval_single,
            group_by=["pipeline_ref", "metric"] + (["id_model"] if not multi_pipeline_mode else []),
            in_col="value",
            out_col="value",
            aggregate_func=agg_func_eval_metric,
        )
    else:
        metrics = df_logs_eval_single

    num_triggers = (
        df_logs[df_logs["id"] == PipelineStage.HANDLE_SINGLE_TRIGGER.name]
        .groupby("pipeline_ref")["id"]
        .agg(num_triggers="count")
        .reset_index()
    )

    num_triggers_adjusted = num_triggers.copy().rename(columns={"num_triggers": "value"})
    num_triggers_adjusted["metric"] = "num_triggers"
    merged = pd.concat([metrics, num_triggers_adjusted]) if multi_pipeline_mode else metrics

    multi_metric_fig = px.box(
        merged,
        x="pipeline_ref" if multi_pipeline_mode else "id_model",
        y="value",
        facet_col="metric",
        labels={"pipeline_ref": "Pipeline", "value": "metric value", "metric": "Metric"},
        title="Merged metrics",
        category_orders={
            "pipeline_ref": list(sorted(merged["pipeline_ref"].unique())),
            "id_model": (list(sorted(merged["id_model"].unique())) if not multi_pipeline_mode else []),
            "metric": list(sorted(merged["metric"].unique())),
        },
        facet_col_wrap=2,
        height=300 * len(merged["metric"].unique()),
    )
    multi_metric_fig.update_yaxes(matches=None, showticklabels=True)  # y axis should be independent (different metrics)
    multi_metric_fig.update_xaxes(showticklabels=True)
    return multi_metric_fig


def section4_1d_boxplots(
    page: str,
    multi_pipeline_mode: bool,
    df_all: pd.DataFrame,
    df_eval_single: pd.DataFrame,
    composite_model_variant: CompositeModelOptions,
) -> html.Div:
    assert "pipeline_ref" in list(df_all.columns)
    assert "pipeline_ref" in list(df_eval_single.columns)

    if page not in _shared_data:
        _shared_data[page] = _PageState(
            composite_model_variant=composite_model_variant,
            df_all=df_all,
            df_eval_single=df_eval_single,
        )
    _shared_data[page].df_all = df_all
    _shared_data[page].df_eval_single = df_eval_single

    @callback(
        Output(f"{page}-1d-box-plot-metrics", "figure"),
        Input(f"{page}-radio-1d-evaluation-handler", "value"),
        Input(f"{page}-radio-1d-dataset-id", "value"),
        Input(f"{page}-radio-1d-eval-metric-agg-func", "value"),
        Input(f"{page}-radio-1d-eval-metric-only-active-model-periods", "value"),
    )
    def update_scatter_num_triggers(
        eval_handler_ref: str,
        dataset_id: str,
        agg_func_eval_metric: OPTIONAL_EVAL_AGGREGATION_FUNCTION,
        only_active_periods: bool = True,
    ) -> go.Figure:
        return gen_figs_1d_eval(
            page,
            multi_pipeline_mode,
            eval_handler_ref,
            dataset_id,
            agg_func_eval_metric,
            only_active_periods,
        )

    # DATA (bring all metrics into columns of one dataframe)

    eval_handler_refs = list(df_eval_single["eval_handler"].unique())
    eval_datasets = list(df_eval_single["dataset_id"].unique())

    return html.Div(
        [
            dcc.Markdown(
                """
        ### Single metric comparison

        _Select the metric to plot against the number of triggers._
    """
            ),
            (
                dcc.Graph(
                    id=f"{page}-1d-box-plot-cost",
                    figure=gen_fig_1d_cost(page),
                )
                if multi_pipeline_mode
                else html.Div([])
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
                                id=f"{page}-radio-1d-evaluation-handler",
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
                                id=f"{page}-radio-1d-dataset-id",
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
                                #### Agg. evaluation metrics
                                over all evaluations
                            """
                            ),
                            dcc.RadioItems(
                                id=f"{page}-radio-1d-eval-metric-agg-func",
                                options=get_args(OPTIONAL_EVAL_AGGREGATION_FUNCTION),
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
                                #### Only active model periods
                                Aggregate only evaluation result in the period where the model was active.
                            """
                            ),
                            dcc.RadioItems(
                                id=f"{page}-radio-1d-eval-metric-only-active-model-periods",
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
                id=f"{page}-1d-box-plot-metrics",
                figure=gen_figs_1d_eval(page, multi_pipeline_mode, eval_handler_refs[0], eval_datasets[0], "sum"),
            ),
        ]
    )
