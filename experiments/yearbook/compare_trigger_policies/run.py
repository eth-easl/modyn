from dataclasses import dataclass
import os
from typing import Literal
from experiments.utils.experiment_runner import run_multiple_pipelines
from experiments.yearbook.compare_trigger_policies.pipeline_config import gen_pipeline_config
from modyn.config.schema.pipeline import DataAmountTriggerConfig, ModynPipelineConfig, TimeTriggerConfig
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig
from modyn.config.schema.pipeline.evaluation.strategy.between_two_triggers import BetweenTwoTriggersEvalStrategyConfig
from modyn.config.schema.pipeline.evaluation.strategy.periodic import PeriodicEvalStrategyConfig
from modyn.config.schema.pipeline.evaluation.strategy.slicing import SlicingEvalStrategyConfig
from modyn.config.schema.pipeline.trigger import DataDriftTriggerConfig
from modyn.utils.utils import SECONDS_PER_UNIT
from modynclient.config.schema.client_config import ModynClientConfig, Supervisor


@dataclass
class Experiment:
    name: str
    eval_handlers: list[EvalHandlerConfig]
    time_trigger_schedules: list[int]  # in years
    data_amount_triggers: list[int]  # in num samples
    drift_triggers: list[tuple[int, float]]  # interval, threshold


_first_timestamp = 0
_last_timestamp = SECONDS_PER_UNIT["d"] * (2013 - 1930)

def construct_slicing_eval_handler() -> EvalHandlerConfig:
    return EvalHandlerConfig(
            name="slice-matrix",
            execution_time="after_pipeline",
            models="matrix",
            strategy=SlicingEvalStrategyConfig(
                eval_every="1d",
                eval_start_from=_first_timestamp,
                eval_end_at=_last_timestamp
            ),
            datasets=["yearbook_test"],
        )

def construct_periodic_eval_handlers(intervals: list[tuple[str, str]]) -> list[EvalHandlerConfig]:
    """
    Args:
        intervals: List of (handler_name_suffix, interval string expression)
    """
    return [
        EvalHandlerConfig(
            name=f"scheduled-{interval}",
            execution_time="manual",
            models="matrix",
            strategy=PeriodicEvalStrategyConfig(
                every="1d",  # every year
                interval=f"[-{fake_interval}; +{fake_interval}]",
                start_timestamp=_first_timestamp,
                end_timestamp=_last_timestamp,
            ),
            datasets=["yearbook_test"],
        )
        for (interval, fake_interval) in []
    ]

def construct_between_trigger_eval_handler() -> EvalHandlerConfig:
    return EvalHandlerConfig(
        name="full",
        execution_time="manual",
        models="most_recent",
        strategy=BetweenTwoTriggersEvalStrategyConfig(),
        datasets=["yearbook", "yearbook_test"],
    )

def construct_pipelines(experiment: Experiment) -> list[ModynPipelineConfig]:
    pipeline_configs: list[ModynPipelineConfig] = []

    # time based triggers: every:
    for years in experiment.time_trigger_schedules:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"timetrigger_{years}y",
                trigger=TimeTriggerConfig(every=f"{years}d", start_timestamp=_first_timestamp),  # faked timestamps
                eval_handlers=experiment.eval_handlers,
            )
        )

    # sample count based triggers: every: 100, 500, 1000, 2000, 10_000
    for count in experiment.data_amount_triggers:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"dataamounttrigger_{count}",
                trigger=DataAmountTriggerConfig(num_samples=count),
                eval_handlers=experiment.eval_handlers,
            )
        )

    for interval, threshold in experiment.drift_triggers:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"datadrifttrigger_{interval}_{threshold}",
                trigger=DataDriftTriggerConfig(
                    detection_interval_data_points=interval,
                    sample_size=None,
                    metric="model",
                    metric_config={"threshold": threshold},
                ),
                eval_handlers=experiment.eval_handlers,
            )
        )

    return pipeline_configs

_EXPERIMENT_REFS = {
    0: Experiment(
        # to verify online composite modle determination logic
        name="timetrigger-smoke-test",
        eval_handlers=[],
        time_trigger_schedules=[],
        data_amount_triggers=[],
        drift_triggers=[]

    ),
    1: Experiment(
        name="training-time-vs-numsamples"
        eval_handlers=[],
        time_trigger_schedules=[],
        data_amount_triggers=[],
        drift_triggers=[]
    ),
    2: Experiment(
        name="full-todo",
        eval_handlers=[],
        time_trigger_schedules=[1, 2, 3, 5, 15, 25, 40],
        data_amount_triggers=[100, 500, 1000, 2000, 10_000],
        drift_triggers=[
            (500, 0.7),
            (1_000, 0.5),
            (1_000, 0.6),
            (1_000, 0.7),
            (1_000, 0.8),
            (1_000, 0.9),
            (5_000, 0.7),
            (10_000, 0.7),
        ]
    )
}

def run_experiment() -> None:
    host = os.getenv("MODYN_SUPERVISOR_HOST")
    port = os.getenv("MODYN_SUPERVISOR_PORT")

    if not host:
        host = input("Enter the supervisors host address: ") or "localhost"
    if not port:
        port = int(input("Enter the supervisors port: ")) or 50063

    # eval_handlers = construct_periodic_eval_handlers([("~1y", "300d"), ("1y", "1d"), ("2y", "2d"), ("3y", "3d"), ("5y", "5d")])

    run_multiple_pipelines(
        client_config=ModynClientConfig(supervisor=Supervisor(ip=host, port=port)),
        pipeline_configs=construct_pipelines(),
        start_replay_at=0,
        stop_replay_at=None,
        maximum_triggers=None,
    )


if __name__ == "__main__":
    run_experiment()
