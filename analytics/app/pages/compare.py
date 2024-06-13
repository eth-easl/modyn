import dash
import pandas as pd
from analytics.app.data.load import list_pipelines, load_pipeline_logs
from analytics.app.data.transform import (
    add_pipeline_ref,
    dfs_models_and_evals,
    leaf_stages,
    logs_dataframe,
    logs_dataframe_agg_by_stage,
)
from analytics.app.pages.plots.cost_over_time import section1_stacked_bar
from analytics.app.pages.plots.eval_heatmap import section_evalheatmap
from analytics.app.pages.plots.eval_over_time import section_metricovertime
from dash import Input, Output, callback, dcc, html

from .plots.cost_vs_eval_metric_agg import section3_scatter_cost_eval_metric
from .plots.num_triggers_eval_metric import section3_scatter_num_triggers
from .plots.one_dimensional_comparison import section4_1d_boxplots

dash.register_page(__name__, path="/compare", title="Pipeline Comparison")

pipelines = list_pipelines()

# -------------------------------------------------------------------------------------------------------------------- #
#                                                         PAGE                                                         #
# -------------------------------------------------------------------------------------------------------------------- #

pipelines = list_pipelines()
initial_pipeline_ids = list(sorted(pipelines.keys()))[:1]


@callback(Output("pipelines-info", "children"), Input("pipelines-selector", "value"))
def switch_pipelines(pipeline_ids: list[int]):
    return render_pipeline_infos(pipeline_ids)


ui_pipelines_selection = html.Div(
    [
        dcc.Markdown("## Select pipeline to analyze"),
        dcc.Dropdown(
            id="pipelines-selector",
            options=[
                {"label": f"{pipeline_id}: {pipeline_name}", "value": pipeline_id}
                for pipeline_id, (pipeline_name, _) in pipelines.items()
            ],
            value=initial_pipeline_ids,
            multi=True,
            clearable=False,
            persistence=True,
            style={"color": "black"},
        ),
    ]
)


def render_pipeline_infos(pipeline_ids: list[int]) -> list[html.Div]:
    # --------------------------------------------------- DATA --------------------------------------------------- #

    pipeline_refs = {pipeline_id: f"{pipeline_id} - {pipelines[pipeline_id][0]}" for pipeline_id in pipeline_ids}

    log_list = {pipeline_id: load_pipeline_logs(pipeline_id) for pipeline_id in pipeline_ids}
    df_logs_dict = {
        pipeline_id: add_pipeline_ref(logs_dataframe(logs), pipeline_refs[pipeline_id])
        for pipeline_id, logs in log_list.items()
    }

    pipeline_leaf_stages = {leaf for log in log_list.values() for leaf in leaf_stages(log)}
    df_logs = pd.concat(df_logs_dict.values())
    df_logs_leaf = df_logs[df_logs["id"].isin(pipeline_leaf_stages)]

    df_logs_agg = pd.concat([logs_dataframe_agg_by_stage(df_log) for pipeline_id, df_log in df_logs_dict.items()])
    df_logs_agg_leaf = df_logs_agg[df_logs_agg["id"].isin(pipeline_leaf_stages)]

    _dfs_models_evals: list[str, tuple[str, pd.DataFrame, pd.DataFrame | None]] = [
        (pipeline_refs[pipeline_id], *dfs_models_and_evals(logs, df_logs["sample_time"].max()))
        for pipeline_id, logs in log_list.items()
    ]

    df_logs_models = pd.concat(
        [add_pipeline_ref(single_df_models, pipeline_ref) for pipeline_ref, single_df_models, _ in _dfs_models_evals]
    )

    df_logs_eval_single = pd.concat(
        [
            add_pipeline_ref(_single_eval_df, pipeline_ref)
            for pipeline_ref, _, _single_eval_df in _dfs_models_evals
            if _single_eval_df is not None
        ]
    )

    # -------------------------------------------------- LAYOUT -------------------------------------------------- #

    eval_items = []
    if df_logs_eval_single is None or df_logs_agg is None:
        eval_items.append(
            dcc.Markdown(
                """
                ## Evaluation metrics missing

                Please run the pipeline with evaluation metrics enabled to enable these evaluation plots.
            """
            )
        )
    else:
        eval_items.append(
            section_metricovertime("compare", True, df_logs_eval_single),
        )
        eval_items.append(section_evalheatmap("compare", True, df_logs_eval_single, df_logs_models))
        eval_items.append(
            section3_scatter_num_triggers("compare", True, df_logs_agg, df_logs_eval_single),
        )
        eval_items.append(
            section3_scatter_cost_eval_metric("compare", df_logs, df_logs_agg_leaf, df_logs_eval_single),
        )
        eval_items.append(section4_1d_boxplots("compare", True, df_logs, df_logs_eval_single))

    return [
        html.H1("Cost over time comparison"),
        section1_stacked_bar("compare", df_logs_leaf),
        html.Div(children=eval_items),
    ]


layout = html.Div(
    [
        dcc.Markdown(
            """
        # Modyn Pipeline Comparison

        To see how different triggers and other pipeline options affect the pipeline performance, we can compare
        different pipelines side by side.
    """
        ),
        ui_pipelines_selection,
        html.Div(id="pipelines-info", children=render_pipeline_infos(initial_pipeline_ids)),
    ]
)
