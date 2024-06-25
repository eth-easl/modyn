import os

import pandas as pd

from experiments.huffpost.compare_trigger_policies.pipeline_config import gen_pipeline_config
from experiments.utils.experiment_runner import run_multiple_pipelines
from modyn.config.schema.pipeline import (
    DataAmountTriggerConfig,
    EvalHandlerConfig,
    ModynPipelineConfig,
    TimeTriggerConfig,
)
from modyn.config.schema.pipeline.evaluation.strategy.slicing import SlicingEvalStrategyConfig
from modynclient.config.schema.client_config import ModynClientConfig, Supervisor


def construct_pipelines() -> list[ModynPipelineConfig]:
    pipeline_configs: list[ModynPipelineConfig] = []
    first_timestamp = pd.to_datetime("2012-01-28").timestamp()
    last_timestamp = pd.to_datetime("2022-09-24").timestamp()

    eval_handlers = [
        EvalHandlerConfig(
            name="slice-matrix",
            execution_time="after_pipeline",
            models="matrix",
            strategy=SlicingEvalStrategyConfig(
                eval_every="1y", eval_start_from=first_timestamp, eval_end_at=last_timestamp
            ),
            datasets=["huffpost_kaggle_test"],
        )
        # for interval in ["1y"] # ["183d", "1y", "2y"]
    ]

    # time based triggers: every: 4y, 3y, 2y, 1y, 183d
    for years in ["1y"]:  # ["4y", "3y", "2y", "1y", "183d"]:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"timetrigger_{years}",
                trigger=TimeTriggerConfig(every=f"{years}"),
                eval_handlers=eval_handlers,
            )
        )

    # sample count based triggers: every: 20_000, 10_000, 5_000, 2_500, 1_000, 500, 200
    for count in []:  # [20_000, 10_000, 5_000, 2_500, 1_000, 500, 200]:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"dataamounttrigger_{count}",
                trigger=DataAmountTriggerConfig(num_samples=count),
                eval_handlers=eval_handlers,
            )
        )

    return pipeline_configs


def run_experiment() -> None:
    host = os.getenv("MODYN_SUPERVISOR_HOST")
    port = int(os.getenv("MODYN_SUPERVISOR_PORT", "0"))

    if not host:
        host = input("Enter the supervisors host address: ") or "localhost"
    if not port:
        port = int(input("Enter the supervisors port: ") or "50063")

    run_multiple_pipelines(
        client_config=ModynClientConfig(supervisor=Supervisor(ip=host, port=port)),
        pipeline_configs=construct_pipelines(),
        start_replay_at=0,
        stop_replay_at=None,
        maximum_triggers=None,
    )


if __name__ == "__main__":
    run_experiment()
