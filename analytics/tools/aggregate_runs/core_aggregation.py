import logging
from copy import deepcopy
from pathlib import Path

import pandas as pd

from analytics.app.data.transform import dfs_models_and_evals, logs_dataframe
from analytics.tools.aggregate_runs.dir_utils import load_multiple_logfiles
from analytics.tools.aggregate_runs.pipeline_equivalence import assert_pipeline_equivalence
from modyn.supervisor.internal.grpc.enums import PipelineStage
from modyn.supervisor.internal.pipeline_executor.models import MultiEvaluationInfo, PipelineLogs

DEBUGGING_MODE = False
"""If True, the the process will halt on breakpoints to allow for manual
verification."""


def merge_files_for_equivalence_group(pipeline_files: list[Path], output_directory: Path) -> None:
    """Merges the logfiles of a group of equivalent pipelines into one file."""
    logs = load_multiple_logfiles(pipeline_files)
    assert_pipeline_equivalence(logs)

    dfs_logs = [logs_dataframe(log) for log in logs]

    max_sample_time = max([df["sample_time"].max() for df in dfs_logs])

    dfs_models_evals: list[tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]] = [
        dfs_models_and_evals(log, max_sample_time) for log in logs
    ]

    df_models = pd.concat([_df_models for _df_models, _, _ in dfs_models_evals])
    assert df_models.shape[0] > 0

    df_eval_requests = pd.concat(
        [
            single_df_eval_requests
            for _, single_df_eval_requests, _ in dfs_models_evals
            if single_df_eval_requests is not None
        ]
    )
    assert df_eval_requests.shape[0] > 0

    df_eval_single = pd.concat(
        [_single_eval_df for _, _, _single_eval_df in dfs_models_evals if _single_eval_df is not None]
    )

    if DEBUGGING_MODE:
        # TEMPLATE
        # df_eval_single[
        #     (df_eval_single["model_idx"] == 1)
        #     & (df_eval_single["eval_handler"] == "exactmatrix")  # ADJUST THIS
        #     & (df_eval_single["dataset_id"] == "cglm_landmark_min25-test")  # ADJUST THIS
        #     & (df_eval_single["interval_start"] == "2004-01-01")  # ADJUST THIS
        #     & (df_eval_single["interval_end"] == "2004-12-31")  # ADJUST THIS
        #     & (df_eval_single["metric"] == "Accuracy")
        # ]
        breakpoint()

    aggregated_logs = aggregate_eval_metrics(df_eval_single, logs)
    aggregated_logs.materialize(output_directory, mode="final")

    if DEBUGGING_MODE:
        breakpoint()


def aggregate_eval_metrics(df_eval_single: pd.DataFrame, logs: list[PipelineLogs]) -> PipelineLogs:
    """Aggregates the evaluation metrics group-wise and updates the creates a
    new PipelineLogs object using the first log in the list as a template."""

    # --------------------------------------- Aggregation within eval dataframe -------------------------------------- #
    groups = df_eval_single.groupby(
        ["model_idx", "eval_handler", "dataset_id", "interval_start", "interval_end", "metric"]
    )

    sizes = groups.agg(size=("model_idx", "size")).reset_index()
    if len(sizes["size"].unique()) != 1 or int(sizes["size"].unique()[0]) != len(logs):
        logging.warning(f"\n{sizes[sizes['size'] != len(logs)]}")
        logging.warning(
            "The number of records in every group is not equal to the number of logs. "
            "This might be due to missing records in the logs or a wrong grouping primary key. "
            "If only a few records show less than the expected number of logs, you might want to "
            "ignore and continue by pressing any key."
        )
        breakpoint()

    aggregated_metrics = groups.agg(
        agg_value=("value", "mean"), id_model_list=("id_model", lambda x: list(x))
    ).reset_index()

    # sanity check: per aggregated row we find len(logs) unique id_model
    agg = aggregated_metrics[["model_idx", "id_model_list"]]
    agg["num_models"] = agg["id_model_list"].apply(len)
    breaking_rows = agg[agg["num_models"] != len(logs)]
    if breaking_rows.shape[0] > 0:
        logging.warning(f"\n{breaking_rows}")
        logging.warning(
            "The number of unique id_model in the aggregated metrics is not equal to the number of logs. Please verify."
        )
        breakpoint()

    if DEBUGGING_MODE:
        # print(aggregated_metrics[["model_idx", "id_model_list"]])
        breakpoint()

    # ---------------------------------- Write back dataframe to PipelineLogs object --------------------------------- #

    aggregated_logs = deepcopy(logs[0])
    for log in aggregated_logs.supervisor_logs.stage_runs:
        if log.id == PipelineStage.EVALUATE_MULTI.name:
            assert isinstance(log.info, MultiEvaluationInfo)

            for single_eval in log.info.interval_results:
                if not single_eval.results:
                    continue

                eval_req = single_eval.eval_request

                # will yield multiple rows (one per each metric)
                request_lookup = aggregated_metrics[
                    (aggregated_metrics["id_model_list"].apply(lambda x: eval_req.id_model in x))
                    & (aggregated_metrics["eval_handler"] == eval_req.eval_handler)
                    & (aggregated_metrics["dataset_id"] == eval_req.dataset_id)
                    & (aggregated_metrics["interval_start"] == pd.to_datetime(eval_req.interval_start, unit="s"))
                    & (aggregated_metrics["interval_end"] == pd.to_datetime(eval_req.interval_end, unit="s"))
                ]

                # find aggregated value
                for metric in single_eval.results["metrics"]:
                    lookup = request_lookup[request_lookup["metric"] == metric["name"]]
                    assert len(lookup) == 1, f"Primary key not unique: {metric['name']}"
                    metric["result"] = float(lookup["agg_value"].iloc[0])

    return aggregated_logs
