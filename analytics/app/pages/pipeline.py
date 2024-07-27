import dash
from analytics.app.data.const import CompositeModelOptions
from analytics.app.pages.const.text import COMPOSITE_MODEL_TEXT
from analytics.app.pages.plots.eval_heatmap import section_evalheatmap
from analytics.app.pages.plots.eval_over_time import section_metricovertime
from analytics.app.pages.plots.num_samples import section_num_samples
from analytics.app.pages.plots.one_dimensional_comparison import section4_1d_boxplots
from dash import Input, Output, callback, dcc, html
from typing_extensions import get_args

from .plots.cost_over_time import section_cost_over_time
from .plots.num_triggers_eval_metric import section3_scatter_num_triggers
from .plots.pipeline_info import section0_pipeline
from .state import pipeline_data, pipelines, process_pipeline_data

dash.register_page(__name__, path="/", title="Pipeline Evaluation")

initial_pipeline_id = min(pipelines.keys())

# -------------------------------------------------------------------------------------------------------------------- #
#                                                         PAGE                                                         #
# -------------------------------------------------------------------------------------------------------------------- #


@callback(
    Output("pipeline-info", "children"),
    Input("pipeline-selector", "value"),
    Input("composite-model-variant", "value"),
)
def switch_pipeline(pipeline_id: int, composite_model_variant: CompositeModelOptions) -> list[html.Div]:
    return render_pipeline_info(pipeline_id, composite_model_variant)


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


def render_pipeline_info(pipeline_id: int, composite_model_variant: CompositeModelOptions) -> list[html.Div]:
    # ----------------------------------------------------- DATA ----------------------------------------------------- #

    if pipeline_id not in pipeline_data:
        pipeline_data[pipeline_id] = process_pipeline_data(pipeline_id)

    data = pipeline_data[pipeline_id]

    # ---------------------------------------------------- LAYOUT ---------------------------------------------------- #

    eval_items = []
    if data.df_eval_single is None or data.df_agg is None:
        eval_items.append(
            dcc.Markdown(
                """
                ## Evaluation metrics missing

                Please run the pipeline with evaluation metrics enabled to enable these evaluation plots.
            """
            )
        )
    else:
        eval_items.append(section_metricovertime("pipeline", False, data.df_eval_single, composite_model_variant))
        eval_items.append(
            section_evalheatmap(
                "pipeline",
                False,
                data.df_models,
                data.df_eval_single,
                composite_model_variant,
            )
        )
        eval_items.append(
            section_num_samples(
                "pipeline",
                False,
                data.df_models,
                data.df_eval_requests,
                composite_model_variant,
            )
        )
        eval_items.append(
            section3_scatter_num_triggers(
                "pipeline",
                False,
                data.df_agg,
                data.df_eval_single,
                composite_model_variant,
            )
        )
        eval_items.append(
            section4_1d_boxplots(
                "pipeline",
                False,
                data.df_all,
                data.df_eval_single,
                composite_model_variant,
            )
        )

    return [
        html.Div(
            [
                section0_pipeline(data.logs, data.df_all, data.df_agg_leaf, data.df_add_parents),
                dcc.Markdown(
                    """
                        ## Cost-/Accuracy triggering tradeoff

                        Considers every batch and compare batch costs (trigger evaluation + training) with batch accuracy (= Accuracy of model at the time of the batch).

                        We choose to compare at batch level as on trigger level we always have a freshly trained model.
                        To evaluate the pipeline's accuracy we also have to consider poi nts in time where no trigger was
                        executed (i.e. batches without triggers).
                    """
                ),
                section_cost_over_time("pipeline", data.df_leaf),
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
        html.Div(id="pipeline-info", children=render_pipeline_info(initial_pipeline_id, "currently_active_model")),
    ]
)
