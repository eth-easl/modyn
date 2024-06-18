import dash_cytoscape as cyto
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html
from modyn.supervisor.internal.grpc.enums import PipelineStage
from modyn.supervisor.internal.pipeline_executor.models import PipelineLogs
from plotly import graph_objects as go


def section0_pipeline(
    logs: PipelineLogs, df_logs: pd.DataFrame, df_logs_agg_leaf: pd.DataFrame, df_logs_add_parents: pd.DataFrame
) -> html.Div:
    def gen_stage_duration_histogram(stage_id: str) -> go.Figure:
        return px.histogram(
            df_logs[df_logs["id"] == stage_id],
            title="Stage Duration Histogram",
            hover_data=df_logs.columns,
            marginal="rug",  # rug, box, violin
            x="duration",
            labels={"duration": "duration in seconds", "id": "Pipeline Stage"},
            color="id",
            histfunc="count",
            text_auto=True,
        )

    @callback(Output("pipeline-graph-info", "children"), Input("pipeline-graph", "tapNodeData"))
    def display_tap_node_info(data) -> str:
        if not data or "id" not in data:
            return "Click a node to get more information"
        series_info = df_logs[df_logs["id"] == data["id"]]["duration"].describe().to_string()
        return (
            f"Pipeline Stage: {data['id']}\n"
            f"Number of Runs: {df_logs[df_logs['id'] == data['id']].shape[0]}\n"
            f"Info about pipeline stage duration:\n"
            f"{series_info}"
        )

    # selection callback
    @callback(
        Output("hist-stage-duration", "figure"),
        Input("pipeline-graph", "tapNodeData"),
        prevent_initial_call="initial_duplicate",
    )
    def display_tap_node_duration(data) -> go.Figure:
        if not data or "id" not in data:
            stage_id = PipelineStage.MAIN.name
        else:
            stage_id = data["id"]
        fig_hist_stage_duration = gen_stage_duration_histogram(stage_id)
        return fig_hist_stage_duration

    fig_pie_pipeline = px.pie(
        df_logs_agg_leaf,
        values="sum",
        names="id",
        hole=0.4,
        hover_data=df_logs_agg_leaf,
        custom_data=["max", "min", "mean", "median", "std", "count"],
    )
    # fig_pie_pipeline.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie_pipeline.update_traces(
        hovertemplate=(
            "<b>%{label}</b><br>"
            "call count: %{customdata[0][5]}<br><br>"
            "<b>Duration:</b><br>"
            "<b>sum</b>: %{value} s (plot)<br>"
            "max: %{customdata[0][0]} s<br>"
            "min: %{customdata[0][1]} s<br>"
            "mean: %{customdata[0][2]} s<br>"
            "median: %{customdata[0][3]} s<br>"
            "std: %{customdata[0][4]} s<br>"
        )
    )

    # TODO: add time series from other modyn components

    fig_sunburst = go.Figure(
        go.Sunburst(
            labels=df_logs_add_parents["id"],
            parents=df_logs_add_parents["parent_id"],
            values=df_logs_add_parents["sum"],
        )
    )

    fig_pipeline = cyto.Cytoscape(
        id="pipeline-graph",
        elements=[
            {"data": {"id": stage, "label": f"{stage} [{idx}]"}, "classes": "red" if idx == 0 else "green"}
            for stage, (idx, _) in logs.pipeline_stages.items()
        ]
        + [
            {"data": {"source": p, "target": stage}}
            for stage, (_, parents) in logs.pipeline_stages.items()
            for p in parents
        ],
        layout={"name": "breadthfirst"},
        style={"width": "1000px", "height": "500px"},
        # white labels:
        stylesheet=[
            {
                "selector": "node",
                "style": {
                    "label": "data(label)",
                    # "color": "white",
                },
            },
            # class selectors
            {"selector": ".red", "style": {"background-color": "green"}},
        ],
    )

    pipeline_node_info = html.Pre(
        id="pipeline-graph-info", style={"border": "thin lightgrey solid", "overflowX": "scroll"}
    )

    return html.Div(
        [
            dcc.Markdown(
                """
            ## Pipeline stage information
            ### Pipeline graph (stage hierarchy)
        """
            ),
            fig_pipeline,
            dcc.Markdown(
                """
            ### Pipeline stage information

            _Click a node to get more information about number of runs, time, and other meta information._
        """
            ),
            pipeline_node_info,
            dcc.Graph(id="hist-stage-duration", figure=gen_stage_duration_histogram(PipelineStage.MAIN.name)),
            dcc.Markdown(
                """
            ### Pipeline time breakdown

            Pipeline leaf stages across all sub-trees:
        """
            ),
            dcc.Graph(figure=fig_pie_pipeline),
            dcc.Markdown(
                """
            Hierarchical pipeline sunburst plot
        """
            ),
            dcc.Graph(figure=fig_sunburst),
        ]
    )
