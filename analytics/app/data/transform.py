import datetime
from typing import Any, Literal

import pandas as pd

from modyn.supervisor.internal.grpc.enums import PipelineStage
from modyn.supervisor.internal.pipeline_executor.models import (
    MultiEvaluationInfo,
    PipelineLogs,
    StageLog,
)
from modyn.supervisor.internal.utils.time_tools import (
    generate_real_training_end_timestamp,
)
from modyn.utils.utils import SECONDS_PER_UNIT

AGGREGATION_FUNCTION = Literal["mean", "median", "max", "min", "sum", "std"]
EVAL_AGGREGATION_FUNCTION = Literal["time_weighted_avg", "mean", "median", "max", "min", "sum", "std"]
OPTIONAL_EVAL_AGGREGATION_FUNCTION = Literal["none", "time_weighted_avg", "mean", "median", "max", "min", "sum", "std"]

# -------------------------------------------------------------------------------------------------------------------- #
#                                                   CREATE dataframes                                                  #
# -------------------------------------------------------------------------------------------------------------------- #


def pipeline_leaf_times_df(
    logs: PipelineLogs,
    *,
    use_traintime_patch_at_trainer: bool,
    pipeline_id: str = "pipeline",
) -> pd.DataFrame:
    pipeline_leaf_stages = leaf_stages(logs)
    df_all = logs_dataframe(logs, f"pipeline_{pipeline_id}")
    df_leaf_single = df_all[df_all["id"].isin(pipeline_leaf_stages)]
    if not use_traintime_patch_at_trainer:
        return df_leaf_single

    df_leaf_only_train = df_leaf_single[df_leaf_single["id"] == PipelineStage.TRAIN.name]
    df_leaf_wo_train = df_leaf_single[df_leaf_single["id"] != PipelineStage.TRAIN.name]

    df_trainings = StageLog.df(
        (x for x in logs.supervisor_logs.stage_runs if x.id == PipelineStage.TRAIN.name),
        extended=True,
    )
    df_merged = df_leaf_only_train.merge(df_trainings, on="trigger_idx", how="inner", suffixes=("", "_training"))
    assert df_merged.shape[0] == df_leaf_only_train.shape[0] == df_trainings.shape[0]
    df_merged["duration"] = df_merged["train_time_at_trainer"] / 1000.0  # ms to s
    df_merged = df_merged[df_leaf_only_train.columns]

    return pd.concat([df_merged, df_leaf_wo_train])


def logs_dataframe(logs: PipelineLogs, pipeline_ref: str = "pipeline") -> pd.DataFrame:
    df = logs.supervisor_logs.df
    df["pipeline_ref"] = pipeline_ref
    df["duration"] = df["duration"].apply(lambda x: x.total_seconds())
    convert_epoch_to_datetime(df, "sample_time")
    return df


def logs_dataframe_agg_by_stage(stage_run_df: pd.DataFrame) -> pd.DataFrame:
    df_agg = (
        stage_run_df.groupby((["pipeline_ref"] if "pipeline_ref" in stage_run_df.columns else []) + ["id"])
        .agg(
            max=("duration", "max"),
            min=("duration", "min"),
            mean=("duration", "mean"),
            median=("duration", "median"),
            std=("duration", "std"),
            sum=("duration", "sum"),
            count=("duration", "count"),
        )
        .fillna(-1)
    )
    df_agg.reset_index(inplace=True)
    return df_agg


def pipeline_stage_parents(logs: PipelineLogs) -> pd.DataFrame:
    ids = []
    parents = []
    for i, (_, parent_list) in logs.pipeline_stages.items():
        if len(parent_list) == 1:
            ids.append(i)
            parents.append(parent_list[0])
        if len(parent_list) > 1:
            if i == PipelineStage.PROCESS_NEW_DATA.name:
                if logs.experiment:
                    ids.append(i)
                    parents.append(PipelineStage.REPLAY_DATA.name)
                else:
                    ids.append(i)
                    parents.append(PipelineStage.FETCH_NEW_DATA.name)
            else:
                raise ValueError(f"Stage {i} has multiple parents: {parent_list}")

    df = pd.DataFrame({"id": ids, "parent_id": parents})
    return df


def dfs_models_and_evals(
    logs: PipelineLogs, max_sample_time: Any, pipeline_ref: str = "pipeline"
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Returns a dataframe with the stored models and the dataframe for
    evaluations."""

    # ---------------------------------------------------- MODELS ---------------------------------------------------- #

    # PipelineStage.STORE_TRAINED_MODEL
    df_store_models = StageLog.df(
        (x for x in logs.supervisor_logs.stage_runs if x.id == PipelineStage.STORE_TRAINED_MODEL.name),
        extended=True,
    )
    df_store_models.set_index(["trigger_idx"], inplace=True)

    # PipelineStage.HANDLE_SINGLE_TRIGGER
    df_single_triggers = StageLog.df(
        (x for x in logs.supervisor_logs.stage_runs if x.id == PipelineStage.HANDLE_SINGLE_TRIGGER.name),
        extended=True,
    )[["trigger_idx", "trigger_id", "first_timestamp", "last_timestamp"]]
    df_single_triggers.set_index(["trigger_idx"], inplace=True)

    # PipelineStage.TRAIN
    df_single_trainings = StageLog.df(
        (x for x in logs.supervisor_logs.stage_runs if x.id == PipelineStage.TRAIN.name),
        extended=True,
    )[["trigger_idx", "num_batches", "num_samples"]]
    df_single_trainings.set_index(["trigger_idx"], inplace=True)

    # MERGE
    joined_models = df_store_models.merge(
        df_single_triggers, on="trigger_idx", how="left", suffixes=("", "_trigger")
    ).merge(df_single_trainings, on="trigger_idx", how="left", suffixes=("", "_training"))

    # sort models by trigger_id (we need that for the shift functions in generate_real_training_end_timestamp etc.)
    joined_models.sort_index(level="trigger_idx", inplace=True)

    joined_models["train_start"] = joined_models["first_timestamp"]
    joined_models["train_end"] = joined_models["last_timestamp"]
    joined_models["real_train_end"] = generate_real_training_end_timestamp(joined_models)

    convert_epoch_to_datetime(joined_models, "sample_time")
    convert_epoch_to_datetime(joined_models, "train_start")
    convert_epoch_to_datetime(joined_models, "train_end")
    convert_epoch_to_datetime(joined_models, "real_train_end")

    df_models = joined_models[
        [col for col in df_store_models.columns if col not in joined_models.index.names]
        + ["train_start", "train_end", "real_train_end", "num_batches", "num_samples"]
    ]

    df_models.reset_index(inplace=True)

    # model_usage period
    df_models["usage_start"] = df_models["real_train_end"] + pd.DateOffset(seconds=1)
    df_models["usage_end"] = df_models["real_train_end"].shift(-1)
    df_models["usage_end"] = df_models["usage_end"].fillna(max_sample_time)

    # linearize ids:
    df_models["trigger_idx"] = df_models["trigger_id"]
    df_models["training_idx"] = df_models["training_id"]
    df_models["model_idx"] = df_models["id_model"]
    _, trigger_idx_mappings = linearize_ids(df_models, [], "training_idx")
    _, model_idx_mappings = linearize_ids(df_models, [], "model_idx")

    df_models["pipeline_ref"] = pipeline_ref

    # -------------------------------------------------- EVALUATIONS ------------------------------------------------- #

    dfs_requests = MultiEvaluationInfo.requests_df(logs.supervisor_logs.stage_runs)
    dfs_metrics = MultiEvaluationInfo.results_df(
        [run.info for run in logs.supervisor_logs.stage_runs if isinstance(run.info, MultiEvaluationInfo)]
    )

    if dfs_requests.shape[0] == 0 or dfs_metrics.shape[0] == 0:
        return df_models, None, None

    for evals_df in [dfs_requests, dfs_metrics]:
        evals_df["interval_center"] = (evals_df["interval_start"] + evals_df["interval_end"]) / 2
        convert_epoch_to_datetime(evals_df, "interval_start")
        convert_epoch_to_datetime(evals_df, "interval_end")
        convert_epoch_to_datetime(evals_df, "interval_center")
        evals_df.sort_values(by=["interval_center"], inplace=True)

        # linearize ids:
        evals_df["training_idx"] = evals_df["training_id"]
        evals_df["model_idx"] = evals_df["id_model"]
        linearize_ids(evals_df, [], "training_idx", trigger_idx_mappings)
        linearize_ids(evals_df, [], "model_idx", model_idx_mappings)

    dfs_requests["pipeline_ref"] = pipeline_ref
    dfs_metrics["pipeline_ref"] = pipeline_ref

    return df_models, dfs_requests, dfs_metrics


# -------------------------------------------------------------------------------------------------------------------- #
#                                                 CREATE other objects                                                 #
# -------------------------------------------------------------------------------------------------------------------- #


def leaf_stages(logs: PipelineLogs) -> list[str]:
    referenced_as_parent = set()
    for _, parent_list in logs.pipeline_stages.values():
        for parent in parent_list:
            referenced_as_parent.add(parent)

    return [stage for stage in logs.pipeline_stages if stage not in referenced_as_parent]


# -------------------------------------------------------------------------------------------------------------------- #
#                                                  TRANSFORM dataframe                                                 #
# -------------------------------------------------------------------------------------------------------------------- #

# ---------------------------------------------------- REFERENCES ---------------------------------------------------- #


def add_pipeline_ref(df: pd.DataFrame | None, ref: str) -> pd.DataFrame | None:
    if df is None:
        return None
    df["pipeline_ref"] = ref
    return df


def linearize_ids(
    df: pd.DataFrame,
    group_columns: list[str],
    target_col: str,
    mapping: dict[tuple[str], dict[int, int]] | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str], dict[int, int]]]:
    """Within certain groups of a dataframe we want to linearize the ids
    (model_ids, trigger_ids, ...) so that they are consecutive integers
    starting from 1.

    Args:
        df: DataFrame with the ids.
        group_columns: Columns to group by, linearization is done within these groups.
        target_col: The column that should be linearized.
        mapping: If provided, this mapping will be applied, otherwise or if groups i not covered a new one will be
            created. (outer dict: group, inner dict: old_id -> new_id)

    Returns:
        DataFrame with linearized ids (inplace operation), and the mapping used.
            (outer dict: group, inner dict: old_id -> new_id)
    """
    mapping = mapping or {}

    if group_columns:
        for group, group_df in df.groupby(group_columns):
            if group not in mapping:
                mapping[group] = {old_id: i + 1 for i, old_id in enumerate(sorted(group_df[target_col].unique()))}
            df.loc[group_df.index, target_col] = df.loc[group_df.index, target_col].map(mapping[group])
    else:
        if not mapping:
            mapping[()] = {old_id: i + 1 for i, old_id in enumerate(sorted(df[target_col].unique()))}  # type: ignore
        df[target_col] = df[target_col].map(mapping[()])  # type: ignore

    return df, mapping


# ------------------------------------------------------- TIME ------------------------------------------------------- #


def convert_epoch_to_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert epoch time to datetime in place.

    Args:
        df: DataFrame with epoch time.
        column: Column with epoch time.
    Returns:
        DataFrame with datetime.
    """
    df[column] = pd.to_datetime(df[column], unit="s")
    return df


def patch_yearbook_time(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Patch yearbook time in place.

    Args:
        df: DataFrame with yearbook time.
        column: Column with yearbook time, has to be a datetime column.
    Returns:
        DataFrame with patched yearbook time.
    """
    if df.shape[0] == 0:
        df[column] = pd.to_datetime([])
        return df
    delta = df[column] - pd.to_datetime("1970-01-01")
    partial_years = delta.dt.seconds / SECONDS_PER_UNIT["d"]
    partial_years_delta = partial_years.apply(lambda x: datetime.timedelta(seconds=x * SECONDS_PER_UNIT["y"]))
    df[column] = pd.to_datetime(delta.apply(lambda x: f"{1930 + x.days}-01-01")) + partial_years_delta
    return df


def df_aggregate_eval_metric(
    df: pd.DataFrame,
    group_by: list[str],
    in_col: str,
    out_col: str,
    aggregate_func: EVAL_AGGREGATION_FUNCTION,
    interval_start: str = "interval_start",
    interval_end: str = "interval_end",
) -> pd.DataFrame:
    """Aggregate evaluation metrics over time.

    Args:
        df: DataFrame with evaluation metrics.
        group_by: Columns to group by.
        in_col: Column with evaluation metrics.
        out_col: Column to store the aggregated evaluation metrics.
        aggregate_func:
        interval_start: Start of the interval.
        interval_end: End of the interval.
    """
    if aggregate_func == "time_weighted_avg":
        # Compute the duration (end - start) as the weight
        df["weight"] = df[interval_end] - df[interval_start]
        group_total_weights = df.groupby(group_by)["weight"].agg(weight_sum="sum")
        group_total_weights.reset_index(inplace=True)

        # Compute the weighted value
        df["weighted_value"] = df[in_col] * df["weight"]

        # Group by `group_by` and compute the weighted average
        grouped = df.groupby(group_by)["weighted_value"].agg(sum_weighted_value="sum")
        grouped.reset_index(inplace=True)

        # add weightsum info
        grouped = grouped.merge(group_total_weights, on=group_by)
        grouped[out_col] = grouped["sum_weighted_value"] / grouped["weight_sum"]

        return grouped[[*group_by, out_col]]

    else:
        # normal average
        df = df.groupby(group_by).agg({in_col: aggregate_func})
        df.reset_index(inplace=True)
        return df.rename(columns={in_col: out_col})
