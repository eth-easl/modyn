import os

import pandas as pd

from experiments.arxiv.compare_trigger_policies.pipeline_config import gen_pipeline_config
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

_FIRST_TIMESTAMP = int(pd.to_datetime("1989-10-26").timestamp())
_LAST_TIMESTAMP = int(pd.to_datetime("2024-06-06").timestamp())  # last: dummy


def construct_slicing_eval_handler(slice: str, first_timestamp: int) -> EvalHandlerConfig:
    return EvalHandlerConfig(
        name=f"slice-matrix-{slice}",
        execution_time="after_pipeline",
        models="matrix",
        strategy=SlicingEvalStrategyConfig(
            eval_every=f"{slice}", eval_start_from=first_timestamp, eval_end_at=_LAST_TIMESTAMP
        ),
        datasets=["arxiv_kaggle_test"],
    )


def construct_periodic_eval_handlers(intervals: list[tuple[str, str]], first_timestamp: int) -> list[EvalHandlerConfig]:
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
                every="1d",  # every year
                interval=f"[-{fake_interval}; +{fake_interval}]",
                start_timestamp=first_timestamp,
                end_timestamp=_LAST_TIMESTAMP,
            ),
            datasets=["arxiv_kaggle_test"],
        )
        for (interval, fake_interval) in intervals
    ]


def construct_between_trigger_eval_handler() -> EvalHandlerConfig:
    return EvalHandlerConfig(
        name="full",
        execution_time="after_pipeline",
        models="active",
        strategy=BetweenTwoTriggersEvalStrategyConfig(),
        datasets=["arxiv_kaggle_all"],  # train and test
    )


def construct_pipelines(experiment: Experiment) -> list[ModynPipelineConfig]:
    pipeline_configs: list[ModynPipelineConfig] = []

    for time in experiment.time_trigger_schedules:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"timetrigger_{time}",
                trigger_config=TimeTriggerConfig(
                    every=f"{time}",
                    start_timestamp=experiment.warmup_until or _FIRST_TIMESTAMP,
                ),
                eval_handlers=experiment.eval_handlers,
            )
        )

    for count in experiment.data_amount_triggers:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"dataamounttrigger_{count}",
                trigger_config=DataAmountTriggerConfig(num_samples=count),
                eval_handlers=experiment.eval_handlers,
            )
        )

    for interval in experiment.drift_detection_intervals:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"datadrifttrigger_{interval}",
                trigger=DataDriftTriggerConfig(
                    evaluation_interval_data_points=interval,
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
        name="arxiv-timetrigger-cold-start",
        eval_handlers=[
            construct_slicing_eval_handler("90d", _FIRST_TIMESTAMP),
            # construct_between_trigger_eval_handler()  # TODO: reenable for arxiv_kaggle_all
        ],
        time_trigger_schedules=["90d"],
        data_amount_triggers=[],
        drift_detection_intervals=[],
        drift_trigger_metrics=[],
        gpu_device="cuda:1",
        warmup_until=_FIRST_TIMESTAMP,
    ),
    # cold training startup vs warmup phase
    1: Experiment(
        # to verify online composite model determination logic
        name="arxiv-timetrigger-warm-start",
        eval_handlers=[
            construct_slicing_eval_handler("90d", int(pd.to_datetime("2000-01-01").timestamp())),
            # construct_between_trigger_eval_handler()  # TODO: reenable for arxiv_kaggle_all
        ],
        time_trigger_schedules=["90d"],
        data_amount_triggers=[],
        drift_detection_intervals=[],
        drift_trigger_metrics=[],
        gpu_device="cuda:2",
        warmup_until=int(pd.to_datetime("2000-01-01").timestamp()),
    ),
    2: Experiment(
        name="arxiv-numsamples-training-time",
        eval_handlers=[construct_between_trigger_eval_handler()],
        time_trigger_schedules=[],
        data_amount_triggers=[100_000, 50_000, 25_000, 10_000, 5_000, 2_000, 1_000, 500],
        drift_detection_intervals=[],
        drift_trigger_metrics=[],
        gpu_device="cuda:1",
    ),
    # tbd. arxiv-timetrigger1y-periodic-eval-intervals
    # tbd. arxiv-drift
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
