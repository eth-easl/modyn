import dash
import pandas as pd
from analytics.app.data.const import CompositeModelOptions
from analytics.app.pages.const.text import COMPOSITE_MODEL_TEXT
from analytics.app.pages.plots.cost_over_time import section_cost_over_time
from analytics.app.pages.plots.eval_heatmap import section_evalheatmap
from analytics.app.pages.plots.eval_over_time import section_metricovertime
from analytics.app.pages.plots.num_samples import section_num_samples
from dash import Input, Output, callback, dcc, html
from typing_extensions import get_args

from .plots.cost_vs_eval_metric_agg import section3_scatter_cost_eval_metric
from .plots.num_triggers_eval_metric import section3_scatter_num_triggers
from .plots.one_dimensional_comparison import section4_1d_boxplots
from .state import pipeline_data, pipelines, process_pipeline_data

dash.register_page(__name__, path="/compare", title="Pipeline Comparison")

initial_pipeline_ids = list(sorted(pipelines.keys()))[:1]

# -------------------------------------------------------------------------------------------------------------------- #
#                                                         PAGE                                                         #
# -------------------------------------------------------------------------------------------------------------------- #


@callback(
    Output("pipelines-info", "children"),
    Input("pipelines-selector", "value"),
    Input("composite-model-variant", "value"),
)
def switch_pipelines(pipeline_ids: list[int], composite_model_variant: CompositeModelOptions) -> list[html.Div]:
    return render_pipeline_infos(pipeline_ids, composite_model_variant)


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
        html.Br(),
        dcc.Markdown(COMPOSITE_MODEL_TEXT),
        dcc.RadioItems(
            id="composite-model-variant",
            options=[{"label": variant, "value": variant} for variant in get_args(CompositeModelOptions)],
            value="currently_active_model",
            persistence=True,
        ),
    ]
)


def render_pipeline_infos(pipeline_ids: list[int], composite_model_variant: CompositeModelOptions) -> list[html.Div]:
    # ----------------------------------------------------- DATA ----------------------------------------------------- #

    for pipeline_id in pipeline_ids:
        if pipeline_id not in pipeline_data:
            pipeline_data[pipeline_id] = process_pipeline_data(pipeline_id)

    df_all = pd.concat([pipeline_data[pipeline_id].df_all for pipeline_id in pipeline_ids])
    df_agg = pd.concat([pipeline_data[pipeline_id].df_agg for pipeline_id in pipeline_ids])
    df_leaf = pd.concat([pipeline_data[pipeline_id].df_leaf for pipeline_id in pipeline_ids])
    df_agg = pd.concat([pipeline_data[pipeline_id].df_agg for pipeline_id in pipeline_ids])
    df_agg_leaf = pd.concat([pipeline_data[pipeline_id].df_agg_leaf for pipeline_id in pipeline_ids])
    df_models = pd.concat([pipeline_data[pipeline_id].df_models for pipeline_id in pipeline_ids])
    df_eval_requests = pd.concat(
        [
            pipeline_data[pipeline_id].df_eval_requests
            for pipeline_id in pipeline_ids
            if pipeline_data[pipeline_id].df_eval_requests is not None
        ]
    )
    df_eval_single = pd.concat(
        [
            pipeline_data[pipeline_id].df_eval_single
            for pipeline_id in pipeline_ids
            if pipeline_data[pipeline_id].df_eval_single is not None
        ]
    )

    # -------------------------------------------------- LAYOUT -------------------------------------------------- #

    eval_items = []
    if df_eval_single is None or df_agg is None:
        eval_items.append(
            dcc.Markdown(
                """
                ## Evaluation metrics missing

                Please run the pipeline with evaluation metrics enabled to enable these evaluation plots.
            """
            )
        )
    else:
        eval_items.append(section_metricovertime("compare", True, df_eval_single, composite_model_variant))
        eval_items.append(section_evalheatmap("compare", True, df_models, df_eval_single, composite_model_variant))
        eval_items.append(section_num_samples("compare", True, df_models, df_eval_requests, composite_model_variant))
        eval_items.append(
            section3_scatter_num_triggers("compare", True, df_agg, df_eval_single, composite_model_variant)
        )
        eval_items.append(
            section3_scatter_cost_eval_metric("compare", df_all, df_agg_leaf, df_eval_single, composite_model_variant)
        )
        eval_items.append(section4_1d_boxplots("compare", True, df_all, df_eval_single, composite_model_variant))

    return [
        html.H1("Cost over time comparison"),
        section_cost_over_time("compare", df_leaf),
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
        html.Div(id="pipelines-info", children=render_pipeline_infos(initial_pipeline_ids, "currently_active_model")),
    ]
)
