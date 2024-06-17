import datetime
import os

from experiments.arxiv.compare_trigger_policies.pipeline_config import gen_pipeline_config
from experiments.utils.experiment_runner import run_multiple_pipelines
from modyn.config.schema.pipeline import (
    DataAmountTriggerConfig,
    EvalHandlerConfig,
    ModynPipelineConfig,
    TimeTriggerConfig,
)
from modyn.config.schema.pipeline.evaluation.strategy.between_two_triggers import BetweenTwoTriggersEvalStrategyConfig
from modyn.config.schema.pipeline.evaluation.strategy.periodic import PeriodicEvalStrategyConfig
from modyn.config.schema.pipeline.evaluation.strategy.slicing import SlicingEvalStrategyConfig
from modyn.config.schema.pipeline.trigger import DataDriftTriggerConfig
from modyn.supervisor.internal.eval.strategies.slicing import SlicingEvalStrategy
from modynclient.config.schema.client_config import ModynClientConfig, Supervisor


def construct_pipelines() -> list[ModynPipelineConfig]:
    pipeline_configs: list[ModynPipelineConfig] = []
    first_timestamp = datetime.datetime(1989, 10, 26).timestamp()
    last_timestamp = datetime.datetime(2024, 6, 6).timestamp()

    eval_handlers = [
        EvalHandlerConfig(
            name=f"scheduled-{interval}",
            execution_time="manual",
            models="matrix",
            strategy=PeriodicEvalStrategyConfig(
                every="90d",
                interval=f"[-{interval}; +{interval}]",
                start_timestamp=first_timestamp,
                end_timestamp=last_timestamp,
            ),
            datasets=["arxiv_kaggle_test"],
        )
        for interval in ["90d", "183d", "1y", "2y"]
    ] + [
        EvalHandlerConfig(
            name="full",
            execution_time="manual",
            models="most_recent",
            strategy=BetweenTwoTriggersEvalStrategyConfig(),
            datasets=["arxiv_kaggle", "arxiv_kaggle_test"],
        ),
    ]

    # time based triggers: every: 4y, 3y, 2y, 1y, 183d
    for years in ["20y", "10y", "5y", "3y", "2y", "1y", "183d", "90d"]:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"timetrigger_{years}",
                trigger=TimeTriggerConfig(every=f"{years}"),
                eval_handlers=eval_handlers,
            )
        )

    # sample count based triggers: every: 1_000_000, 500_000, 100_000, 50_000, 10_000
    for count in [1_000_000, 500_000, 100_000, 50_000, 10_000]:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"dataamounttrigger_{count}",
                trigger=DataAmountTriggerConfig(num_samples=count),
                eval_handlers=eval_handlers,
            )
        )
        
    for interval, threshold in [(20_000, 0.5), (100_000, 0.6), (100_000, 0.7), (100_000, 0.8), (100_000, 0.9), (250_000, 0.7), (500_000, 0.7), (1_000_000, 0.7)]:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"datadrifttrigger_{interval}_{threshold}",
                trigger=DataDriftTriggerConfig(
                    detection_interval_data_points=interval,
                    sample_size=5000,
                    metric="model",
                    metric_config={"threshold": threshold},
                ),
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
