from experiments.utils.experiment_runner import run_multiple_pipelines
from experiments.yearbook.compare_trigger_policies.pipeline_config import gen_pipeline_config
from modyn.config.schema.pipeline import DataAmountTriggerConfig, ModynPipelineConfig, TimeTriggerConfig
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig
from modyn.config.schema.pipeline.evaluation.strategy.between_two_triggers import BetweenTwoTriggersEvalStrategyConfig
from modyn.config.schema.pipeline.evaluation.strategy.periodic import PeriodicEvalStrategyConfig
from modyn.utils.utils import SECONDS_PER_UNIT
from modynclient.config.schema.client_config import ModynClientConfig, Supervisor


def construct_pipelines() -> list[ModynPipelineConfig]:
    pipeline_configs: ModynPipelineConfig = []
    first_timestamp = 0
    last_timestamp = SECONDS_PER_UNIT["d"] * (2015 - 1930)

    eval_handlers = [
        EvalHandlerConfig(
            name=f"scheduled-{interval}",
            execution_time="after_pipeline",
            models="matrix",
            strategy=PeriodicEvalStrategyConfig(
                every="1d",  # every year
                interval=f"[-{fake_interval}; +{fake_interval}]",
                start_timestamp=first_timestamp,
                end_timestamp=last_timestamp,
            ),
            datasets=["yearbook_test"],
        )
        for (interval, fake_interval) in [("1y", "1d"), ("2y", "2d"), ("3y", "3d"), ("5y", "5d")]
    ] + [
        EvalHandlerConfig(
            name="full",
            execution_time="after_pipeline",
            models="most_recent",
            strategy=BetweenTwoTriggersEvalStrategyConfig(),
            datasets=["yearbook", "yearbook_test"],
        ),
    ]

    # time based triggers: every: 1y, 3y, 5y, 15y, 25y, 40y
    for years in [1, 3, 5, 15, 25, 40]:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"timetrigger_{years}y",
                trigger=TimeTriggerConfig(every=f"{years}y"),  # faked timestamps
                eval_handlers=eval_handlers,
            )
        )

    # sample count based triggers: every: 100, 500, 1000, 2000, 10_000
    for count in [100, 500, 1000, 2000, 10_000]:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"dataamounttrigger_{count}",
                trigger=DataAmountTriggerConfig(num_samples=count),
                eval_handlers=eval_handlers,
            )
        )

    return pipeline_configs


def run_experiment() -> None:
    host = input("Enter the supervisors host address: ") or "localhost"
    port = int(input("Enter the supervisors port: ")) or 50063
    run_multiple_pipelines(
        client_config=ModynClientConfig(supervisor=Supervisor(ip=host, port=port)),
        pipeline_configs=construct_pipelines(),
        start_replay_at=0,
        stop_replay_at=None,
        maximum_triggers=None,
    )


if __name__ == "__main__":
    run_experiment()
