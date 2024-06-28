import os
from experiments.models import Experiment
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


_FIRST_TIMESTAMP = 0
_LAST_TIMESTAMP = SECONDS_PER_UNIT["d"] * (2014 - 1930)  # 2014: dummy


def construct_slicing_eval_handler() -> EvalHandlerConfig:
    return EvalHandlerConfig(
        name="slice-matrix",
        execution_time="after_pipeline",
        models="matrix",
        strategy=SlicingEvalStrategyConfig(
            eval_every="1d", eval_start_from=_FIRST_TIMESTAMP, eval_end_at=_LAST_TIMESTAMP
        ),
        datasets=["yearbook_test"],
    )


def construct_periodic_eval_handlers(intervals: list[tuple[str, str]]) -> dict[EvalHandlerConfig]:
    """
    Args:
        intervals: List of (handler_name_suffix, interval string expression)
    """
    return [
        EvalHandlerConfig(
            name=f"scheduled-{interval[0]}",
            execution_time="manual",
            models="matrix",
            strategy=PeriodicEvalStrategyConfig(
                every="1d",  # every year
                interval=f"[-{fake_interval}; +{fake_interval}]",
                start_timestamp=_FIRST_TIMESTAMP,
                end_timestamp=_LAST_TIMESTAMP,
            ),
            datasets=["yearbook_test"],
        )
        for (interval, fake_interval) in intervals
    ]


def construct_between_trigger_eval_handler() -> EvalHandlerConfig:
    return EvalHandlerConfig(
        name="full",
        execution_time="after_pipeline",
        models="active",
        strategy=BetweenTwoTriggersEvalStrategyConfig(),
        datasets=["yearbook_all"],  # train and test
    )


def construct_pipelines(experiment: Experiment) -> list[ModynPipelineConfig]:
    pipeline_configs: list[ModynPipelineConfig] = []

    for years in experiment.time_trigger_schedules:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"{experiment.name}_time_{years}y",
                trigger=TimeTriggerConfig(every=f"{years}d", start_timestamp=_FIRST_TIMESTAMP),  # faked timestamps
                eval_handlers=experiment.eval_handlers,
                device=experiment.gpu_device,
            )
        )

    for count in experiment.data_amount_triggers:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"{experiment.name}_dataamount{count}",
                trigger=DataAmountTriggerConfig(num_samples=count),
                eval_handlers=experiment.eval_handlers,
                device=experiment.gpu_device,
            )
        )

    for interval, threshold in experiment.drift_triggers:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"{experiment.name}_drift_{interval}_{threshold}",
                trigger=DataDriftTriggerConfig(
                    detection_interval_data_points=interval,
                    sample_size=None,
                    metric="model",
                    metric_config={"threshold": threshold},
                ),
                eval_handlers=experiment.eval_handlers,
                device=experiment.gpu_device,
            )
        )

    return pipeline_configs


_EXPERIMENT_REFS = {
    # done
    0: Experiment(
        # to verify online composite model determination logic
        name="timetrigger-smoke-test",
        eval_handlers=[construct_slicing_eval_handler(), construct_between_trigger_eval_handler()],
        time_trigger_schedules=[1, 2, 5],
        data_amount_triggers=[],
        drift_triggers=[],
        gpu_device="cuda:0",
    ),
    # unfinished
    1: Experiment(
        name="numsamples-training-time",
        eval_handlers=[construct_between_trigger_eval_handler()],
        time_trigger_schedules=[],
        data_amount_triggers=[100, 200, 500, 1_000, 2_500, 5_000, 10_000, 15_000, 30_000],
        drift_triggers=[],
        gpu_device="cuda:1",
    ),
    # different time triggcer
    2: Experiment(
        name="timetrigger1y-periodic-eval-intervals",
        eval_handlers=[construct_slicing_eval_handler()]
        + construct_periodic_eval_handlers(
            intervals=[
                ("-delta-current", "23h"),
                ("-delta+-1y", "1d"),
                ("-delta+-2y", "2d"),
                ("-delta+-3y", "3d"),
                ("-delta+-5y", "5d"),
                ("-delta+-10y", "10d"),
                ("-delta+-15y", "15d"),
            ]
        ),
        time_trigger_schedules=[1],
        data_amount_triggers=[],
        drift_triggers=[],
        gpu_device="cuda:2",
    ),
    -2: Experiment(
        name="full-todo",
        eval_handlers=[
            construct_periodic_eval_handlers([("~1y", "300d"), ("1y", "1d"), ("2y", "2d"), ("3y", "3d"), ("5y", "5d")])
        ],
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
        ],
        gpu_device="cuda:0",
    ),
}


def run_experiment() -> None:
    host = os.getenv("MODYN_SUPERVISOR_HOST")
    port = os.getenv("MODYN_SUPERVISOR_PORT")

    if not host:
        host = input("Enter the supervisors host address: ") or "localhost"
    if not port:
        port = int(input("Enter the supervisors port: ")) or 50063

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
