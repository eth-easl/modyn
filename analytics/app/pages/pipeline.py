import dash
from analytics.app.data.load import list_pipelines, load_pipeline_logs
from analytics.app.data.transform import (
    add_pipeline_ref,
    dfs_models_and_evals,
    leaf_stages,
    logs_dataframe,
    logs_dataframe_agg_by_stage,
    pipeline_stage_parents,
)
from analytics.app.pages.plots.eval_heatmap import section_evalheatmap
from analytics.app.pages.plots.eval_over_time import section_metricovertime
from analytics.app.pages.plots.num_samples import section_num_samples
from analytics.app.pages.plots.one_dimensional_comparison import section4_1d_boxplots
from dash import Input, Output, callback, dcc, html

from .plots.cost_over_time import section1_stacked_bar
from .plots.num_triggers_eval_metric import section3_scatter_num_triggers
from .plots.pipeline_info import section0_pipeline

dash.register_page(__name__, path="/", title="Pipeline Evaluation")


# -------------------------------------------------------------------------------------------------------------------- #
#                                                         PAGE                                                         #
# -------------------------------------------------------------------------------------------------------------------- #

pipelines = list_pipelines()
initial_pipeline_id = min(pipelines.keys())


@callback(
    Output("pipeline-info", "children"), Input("pipeline-selector", "value"), prevent_initial_call="initial_duplicate"
)
def switch_pipeline(pipeline_id: int):
    return render_pipeline_info(pipeline_id)


ui_pipeline_selection = html.Div(
    [
        dcc.Markdown("## Select pipeline to analyze"),
        dcc.Dropdown(
            id="pipeline-selector",
            options=[
                {"label": f"{pipeline_id}: {pipeline_name}", "value": pipeline_id}
                for pipeline_id, (pipeline_name, _) in pipelines.items()
            ],
            value=initial_pipeline_id,
            clearable=False,
            persistence=True,
            style={"color": "black", "width": "65%"},
        ),
    ]
)


def render_pipeline_info(pipeline_id: int) -> list[html.Div]:
    # ----------------------------------------------------- DATA ----------------------------------------------------- #

    pipeline_ref = f"{pipeline_id} - {pipelines[pipeline_id][1]}"

    logs = load_pipeline_logs(pipeline_id)
    pipeline_leaf_stages = leaf_stages(logs)
    df_logs = logs_dataframe(logs)
    df_logs_leaf = df_logs[df_logs["id"].isin(pipeline_leaf_stages)]

    df_logs_agg = logs_dataframe_agg_by_stage(df_logs)
    df_logs_agg_leaf = df_logs_agg[df_logs_agg["id"].isin(pipeline_leaf_stages)]

    df_parents = pipeline_stage_parents(logs)
    df_logs_add_parents = df_logs_agg.merge(df_parents, left_on="id", right_on="id", how="left")

    df_logs_models, df_logs_eval_requests, df_logs_eval_single = dfs_models_and_evals(
        logs, df_logs["sample_time"].max()
    )

    # ---------------------------------------------------- LAYOUT ---------------------------------------------------- #

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
            section_metricovertime("pipeline", False, add_pipeline_ref(df_logs_eval_single, pipeline_ref))
        )
        eval_items.append(
            section_evalheatmap(
                "pipeline",
                False,
                add_pipeline_ref(df_logs_eval_single, pipeline_ref),
                add_pipeline_ref(df_logs_models, pipeline_ref),
            )
        )
        eval_items.append(
            section_num_samples(
                "pipeline",
                False,
                add_pipeline_ref(df_logs_models, pipeline_ref),
                add_pipeline_ref(df_logs_eval_requests, pipeline_ref),
            )
        )
        eval_items.append(
            section3_scatter_num_triggers(
                "pipeline",
                False,
                add_pipeline_ref(df_logs_agg, pipeline_ref),
                add_pipeline_ref(df_logs_eval_single, pipeline_ref),
            )
        )
        eval_items.append(
            section4_1d_boxplots(
                "pipeline",
                False,
                add_pipeline_ref(df_logs, pipeline_ref),
                add_pipeline_ref(df_logs_eval_single, pipeline_ref),
            )
        )

    return [
        html.Div(
            [
                section0_pipeline(logs, df_logs, df_logs_agg_leaf, df_logs_add_parents),
                dcc.Markdown(
                    """
                        ## Cost-/Accuracy triggering tradeoff

                        Considers every batch and compare batch costs (trigger evaluation + training) with batch accuracy (= Accuracy of model at the time of the batch).

                        We choose to compare at batch level as on trigger level we always have a freshly trained model.
                        To evaluate the pipeline's accuracy we also have to consider poi nts in time where no trigger was
                        executed (i.e. batches without triggers).
                    """
                ),
                section1_stacked_bar("pipeline", add_pipeline_ref(df_logs_leaf, pipeline_ref)),
                html.Div(eval_items),
            ]
        )
    ]


layout = html.Div(
    [
        dcc.Markdown(
            """
        # Modyn Pipeline Evaluation

        Let's inspect a pipeline run and compare it's cost and accuracy.
        Doing so we can determine weather trigger policies are effective.
    """
        ),
        ui_pipeline_selection,
        html.Div(id="pipeline-info", children=render_pipeline_info(initial_pipeline_id)),
    ]
)
