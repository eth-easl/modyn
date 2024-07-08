import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from analytics.app.data.transform import dfs_models_and_evals, logs_dataframe
from modyn.supervisor.internal.grpc.enums import PipelineStage
from modyn.supervisor.internal.pipeline_executor.models import MultiEvaluationInfo, PipelineLogs, SingleEvaluationInfo


def patch_logfile(log: Path) -> PipelineLogs:
    """
    Converts a logfile to the new batched evaluation format.
    """

    logdict = json.loads(log.read_text(encoding="utf-8"))
    stage_runs = logdict["supervisor_logs"]["stage_runs"]

    new_stage_runs: list[dict] = []

    for stage_log in stage_runs:
        if not stage_log.get("id_seq_num"):
            stage_log["id_seq_num"] = -1
        if stage_log["id"] == "EVALUATE_SINGLE":
            stage_log["id"] = PipelineStage.EVALUATE_MULTI.name
            if stage_log["info"] and stage_log["info"]["eval_request"]:
                # old format
                new_info = MultiEvaluationInfo(
                    dataset_id=stage_log["info"]["eval_request"]["dataset_id"],
                    id_model=stage_log["info"]["eval_request"]["id_model"],
                    interval_results=[
                        SingleEvaluationInfo(
                            eval_request=stage_log["info"]["eval_request"],
                            results=stage_log["info"]["results"],
                            failure_reason=stage_log["info"].get("failure_reason"),
                        )
                    ],
                ).model_dump(by_alias=True)
                stage_log["info"] = new_info
                stage_log["info_type"] = "MultiEvaluationInfo"
                new_stage_runs.append(stage_log)
        else:
            new_stage_runs.append(stage_log)

    logdict["supervisor_logs"]["stage_runs"] = new_stage_runs

    pipeline_logs = PipelineLogs.model_validate(logdict)
    return patch_current_currently_trained_model_dataset_end(pipeline_logs)


def patch_current_currently_trained_model_dataset_end(logs: PipelineLogs) -> PipelineLogs:
    """At the end of the dataset, after the last training there won't be a currently trained model anymore.
    Therefore we will just use the values from the currently active model."""
    pipeline_ref = "pipeline_ref"
    df_all = logs_dataframe(logs, pipeline_ref)
    _, df_eval_requests, _ = dfs_models_and_evals(logs, df_all["sample_time"].max(), pipeline_ref)

    df_eval_requests.sort_values("interval_center", inplace=True)
    # df_eval_requests
    last_model_idx = df_eval_requests["model_idx"].max()
    max_model_evals = df_eval_requests[df_eval_requests["model_idx"] == last_model_idx]

    patched_logs = deepcopy(logs)
    if max_model_evals[max_model_evals["currently_trained_model"]].shape[0] == 0:
        return patched_logs

    max_trained_interval_end = max_model_evals[max_model_evals["currently_trained_model"]]["interval_end"].max()
    last_model_id = max_model_evals[max_model_evals["currently_trained_model"]]["id_model"].max()

    for log in patched_logs.supervisor_logs.stage_runs:
        if log.id == PipelineStage.EVALUATE_MULTI.name:
            assert isinstance(log.info, MultiEvaluationInfo)
            for single_eval in log.info.interval_results:
                if not single_eval.results:
                    continue

                eval_req = single_eval.eval_request
                if (
                    eval_req.id_model == last_model_id
                    and pd.to_datetime(eval_req.interval_end, unit="s") >= max_trained_interval_end
                    and eval_req.currently_active_model
                    and not eval_req.currently_trained_model
                ):
                    eval_req.currently_trained_model = True

    return patched_logs
