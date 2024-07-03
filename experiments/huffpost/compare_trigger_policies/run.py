import os

import pandas as pd
from experiments.huffpost.compare_trigger_policies.pipeline_config import gen_pipeline_config
from experiments.models import Experiment
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
from modyn.config.schema.pipeline.trigger.drift.aggregation import MajorityVoteDriftAggregationStrategy
from modynclient.config.schema.client_config import ModynClientConfig, Supervisor

_FIRST_TIMESTAMP = int(pd.to_datetime("2012-01-28").timestamp())
_LAST_TIMESTAMP = int(pd.to_datetime("2022-09-24").timestamp())  # last: dummy


def construct_slicing_eval_handler(slice: str) -> EvalHandlerConfig:
    return EvalHandlerConfig(
        name=f"slice-matrix-{slice}",
        execution_time="after_pipeline",
        models="matrix",
        strategy=SlicingEvalStrategyConfig(
            eval_every=f"{slice}", eval_start_from=_FIRST_TIMESTAMP, eval_end_at=_LAST_TIMESTAMP
        ),
        datasets=["huffpost_kaggle_test"],
    )


def construct_periodic_eval_handlers(intervals: list[tuple[str, str]]) -> list[EvalHandlerConfig]:
    """
    Args:
        intervals: List of (handler_name_suffix, interval string expression)
    """
    return [
        EvalHandlerConfig(
            name=f"scheduled-{interval}",
            execution_time="after_pipeline",
            models="matrix",
            strategy=PeriodicEvalStrategyConfig(
                every="30d",  # TODO:
                interval=f"[-{fake_interval}; +{fake_interval}]",
                start_timestamp=_FIRST_TIMESTAMP,
                end_timestamp=_LAST_TIMESTAMP,
            ),
            datasets=["huffpost_kaggle_test"],
        )
        for (interval, fake_interval) in intervals
    ]


def construct_between_trigger_eval_handler() -> EvalHandlerConfig:
    return EvalHandlerConfig(
        name="full",
        execution_time="after_pipeline",
        models="active",
        strategy=BetweenTwoTriggersEvalStrategyConfig(),
        datasets=["huffpost_kaggle_all"],  # train and test
    )


def construct_pipelines(experiment: Experiment) -> list[ModynPipelineConfig]:
    pipeline_configs: list[ModynPipelineConfig] = []

    for time in experiment.time_trigger_schedules:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"{experiment.name}_time_{time}",
                trigger=TimeTriggerConfig(every=f"{time}"),
                eval_handlers=experiment.eval_handlers,
            )
        )

    for count in experiment.data_amount_triggers:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"{experiment.name}_dataamount_{count}",
                trigger=DataAmountTriggerConfig(num_samples=count),
                eval_handlers=experiment.eval_handlers,
            )
        )

    for interval in experiment.drift_detection_intervals:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"{experiment.name}_drift_{interval}",
                trigger=DataDriftTriggerConfig(
                    detection_interval_data_points=interval,
                    sample_size=None,
                    metrics=experiment.drift_trigger_metrics,
                    aggregation_strategy=MajorityVoteDriftAggregationStrategy(),
                ),
                eval_handlers=experiment.eval_handlers,
            )
        )

    return pipeline_configs


_EXPERIMENT_REFS = {
    # done
    0: Experiment(
        # to verify online composite model determination logic
        name="huff-timetrigger-smoke-test",
        eval_handlers=[construct_slicing_eval_handler("90d")],
        time_trigger_schedules=["90d"],
        data_amount_triggers=[],
        drift_detection_intervals=[],
        drift_trigger_metrics=[],
        gpu_device="cuda:0",
    ),
    1: Experiment(
        name="hp-numsamples-training-time",
        eval_handlers=[construct_between_trigger_eval_handler()],
        time_trigger_schedules=[],
        data_amount_triggers=[100_000, 50_000, 25_000, 10_000, 5_000, 2_000, 1_000, 500],
        drift_detection_intervals=[],
        drift_trigger_metrics=[],
        gpu_device="cuda:1",
    ),
    # tbd.: huff-timetrigger1y-periodic-eval-intervals
    # tbd.: huff-drift
}


def run_experiment() -> None:
    host = os.getenv("MODYN_SUPERVISOR_HOST")
    port = int(os.getenv("MODYN_SUPERVISOR_PORT", "0"))

    if not host:
        host = input("Enter the supervisors host address: ") or "localhost"
    if not port:
        port = int(input("Enter the supervisors port: ") or "50063")

    experiment_id = int(input("Enter the id of the experiment you want to run: "))

    run_multiple_pipelines(
        client_config=ModynClientConfig(supervisor=Supervisor(ip=host, port=port)),
        pipeline_configs=construct_pipelines(_EXPERIMENT_REFS[experiment_id]),
        start_replay_at=_FIRST_TIMESTAMP,
        stop_replay_at=None,
        maximum_triggers=None,
    )


if __name__ == "__main__":
    run_experiment()
