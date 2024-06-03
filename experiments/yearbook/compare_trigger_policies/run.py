import os

from experiments.utils.experiment_runner import run_multiple_pipelines
from experiments.yearbook.compare_trigger_policies.pipeline_config import gen_pipeline_config
from modyn.config.schema.pipeline import ModynPipelineConfig, TimeTriggerConfig
from modyn.config.schema.pipeline.evaluation import (
    AfterTrainingEvalTriggerConfig,
    EvalHandlerConfig,
    IntervalEvalStrategyConfig,
    PeriodicEvalTriggerConfig,
    UntilNextTriggerEvalStrategyConfig,
)
from modyn.config.schema.pipeline.pipeline import DataAmountTriggerConfig
from modyn.utils.utils import SECONDS_PER_UNIT
from modynclient.config.schema.client_config import ModynClientConfig, Supervisor


def run_experiment() -> None:
    pipeline_configs: list[ModynPipelineConfig] = []

    eval_handlers = [
        EvalHandlerConfig(
            name="scheduled",
            datasets=["yearbook-test"],
            strategy=IntervalEvalStrategyConfig(interval="(-1d, +1d)"),  # timestamps faked (y -> d)
            trigger=PeriodicEvalTriggerConfig(
                every="1d", start_timestamp=0, end_timestamp=(2013 - 1930) * SECONDS_PER_UNIT["d"], matrix=True
            ),
        ),
        EvalHandlerConfig(
            name="scheduled",
            datasets=["yearbook-test"],
            strategy=IntervalEvalStrategyConfig(interval="(-2d, +2d)"),  # timestamps faked (y -> d)
            trigger=PeriodicEvalTriggerConfig(
                every="1d", start_timestamp=0, end_timestamp=(2013 - 1930) * SECONDS_PER_UNIT["d"], matrix=True
            ),
        ),
        EvalHandlerConfig(
            name="scheduled",
            datasets=["yearbook-test"],
            strategy=IntervalEvalStrategyConfig(interval="(-2d, +2d)"),  # timestamps faked (y -> d)
            trigger=PeriodicEvalTriggerConfig(
                every="1d", start_timestamp=0, end_timestamp=(2013 - 1930) * SECONDS_PER_UNIT["d"], matrix=True
            ),
        ),
        EvalHandlerConfig(
            name="full",
            datasets=["yearbook"],
            strategy=UntilNextTriggerEvalStrategyConfig(),
            trigger=AfterTrainingEvalTriggerConfig(),
        ),
    ]

    # time based triggers: every: 1y, 2y, 5y, 15y, 25y, 40y
    for years in [40, 25, 15, 5, 2, 1]:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"timetrigger_{years}y",
                trigger=TimeTriggerConfig(
                    every=f"{years}d",
                ),  # timestamps faked (y -> d)
                eval_handlers=eval_handlers,
            )
        )

    # sample count based triggers: every: 20_000, 10_000, 5_000, 2_500, 1_000, 500
    for count in [20_000, 10_000, 5_000, 2_500, 1_000, 500]:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"dataamounttrigger_{count}",
                trigger=DataAmountTriggerConfig(num_samples=count),
                eval_handlers=eval_handlers,
            )
        )

    host = os.getenv("MODYN_SUPERVISOR_HOST")
    port = os.getenv("MODYN_SUPERVISOR_PORT")

    if not host:
        host = input("Enter the supervisors host address: ") or "localhost"
    if not port:
        port = int(input("Enter the supervisors port: ") or "50063")

    run_multiple_pipelines(
        client_config=ModynClientConfig(supervisor=Supervisor(ip=host, port=port)),
        pipeline_configs=pipeline_configs,
        start_replay_at=0,
        stop_replay_at=None,
        maximum_triggers=None,
    )


if __name__ == "__main__":
    run_experiment()
