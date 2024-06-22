from dataclasses import dataclass

import pandas as pd

from analytics.app.data.load import list_pipelines, load_pipeline_logs
from analytics.app.data.transform import (
    dfs_models_and_evals,
    leaf_stages,
    logs_dataframe,
    logs_dataframe_agg_by_stage,
    pipeline_stage_parents,
)
from modyn.supervisor.internal.pipeline_executor.models import PipelineLogs


@dataclass
class ProcessedPipelineData:
    pipeline_ref: str

    logs: PipelineLogs
    pipeline_leaf_stages: list[str]

    df_all: pd.DataFrame
    df_leaf: pd.DataFrame

    df_agg: pd.DataFrame
    df_agg_leaf: pd.DataFrame

    df_parents: pd.DataFrame
    df_add_parents: pd.DataFrame

    df_models: pd.DataFrame
    df_eval_requests: pd.DataFrame | None
    df_eval_single: pd.DataFrame | None


# ---------------------------------------- Global state (shared by all pages) ---------------------------------------- #

pipelines = list_pipelines()
max_pipeline_id = max(pipelines.keys())

pipeline_data: dict[int, ProcessedPipelineData] = {}


def process_pipeline_data(pipeline_id: int) -> ProcessedPipelineData:
    pipeline_ref = f"{pipeline_id}".zfill(len(str(max_pipeline_id))) + f" - {pipelines[pipeline_id][0]}"

    logs = load_pipeline_logs(pipeline_id)
    pipeline_leaf_stages = leaf_stages(logs)
    df_all = logs_dataframe(logs, pipeline_ref)
    df_leaf = df_all[df_all["id"].isin(pipeline_leaf_stages)]

    df_agg = logs_dataframe_agg_by_stage(df_all)
    df_agg_leaf = df_agg[df_agg["id"].isin(pipeline_leaf_stages)]

    df_parents = pipeline_stage_parents(logs)
    df_add_parents = df_agg.merge(df_parents, left_on="id", right_on="id", how="left")

    df_logs_models, df_eval_requests, df_logs_eval_single = dfs_models_and_evals(
        logs, df_all["sample_time"].max(), pipeline_ref
    )

    return ProcessedPipelineData(
        pipeline_ref=pipeline_ref,
        logs=logs,
        pipeline_leaf_stages=pipeline_leaf_stages,
        df_all=df_all,
        df_leaf=df_leaf,
        df_agg=df_agg,
        df_agg_leaf=df_agg_leaf,
        df_parents=df_parents,
        df_add_parents=df_add_parents,
        df_models=df_logs_models,
        df_eval_requests=df_eval_requests,
        df_eval_single=df_logs_eval_single,
    )
