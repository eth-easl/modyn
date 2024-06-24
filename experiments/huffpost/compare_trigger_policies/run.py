import datetime
import os

from experiments.huffpost.compare_trigger_policies.pipeline_config import gen_pipeline_config
from experiments.utils.experiment_runner import run_multiple_pipelines
from modyn.config.schema.pipeline import (
    DataAmountTriggerConfig,
    EvalHandlerConfig,
    ModynPipelineConfig,
    TimeTriggerConfig,
)
from modyn.config.schema.pipeline.evaluation.strategy.between_two_triggers import BetweenTwoTriggersEvalStrategyConfig
from modyn.config.schema.pipeline.evaluation.strategy.periodic import PeriodicEvalStrategyConfig
from modynclient.config.schema.client_config import ModynClientConfig, Supervisor


def construct_pipelines() -> list[ModynPipelineConfig]:
    pipeline_configs: list[ModynPipelineConfig] = []
    first_timestamp = datetime.datetime(2012, 1, 28).timestamp()
    last_timestamp = datetime.datetime(2022, 9, 24).timestamp()

    eval_handlers = [
        EvalHandlerConfig(
            name=f"scheduled-{interval}",
            execution_time="after_pipeline",
            models="matrix",
            strategy=PeriodicEvalStrategyConfig(
                every="183d",
                interval=f"[-{interval}; +{interval}]",
                start_timestamp=first_timestamp,
                end_timestamp=last_timestamp,
            ),
            datasets=["huffpost_kaggle_test"],
        )
        for interval in ["183d", "1y", "2y"]
    ] + [
        EvalHandlerConfig(
            name="full",
            execution_time="after_pipeline",
            models="active",
            strategy=BetweenTwoTriggersEvalStrategyConfig(),
            datasets=["huffpost_kaggle", "huffpost_kaggle_test"],
        ),
    ]

    # time based triggers: every: 4y, 3y, 2y, 1y, 183d
    for years in ["4y", "3y", "2y", "1y", "183d"]:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"timetrigger_{years}",
                trigger=TimeTriggerConfig(every=f"{years}"),
                eval_handlers=eval_handlers,
            )
        )

    # sample count based triggers: every: 20_000, 10_000, 5_000, 2_500, 1_000, 500, 200
    for count in [20_000, 10_000, 5_000, 2_500, 1_000, 500, 200]:
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
