import os

import pandas as pd

from experiments.arxiv.compare_trigger_policies.pipeline_config import gen_pipeline_config
from experiments.utils.models import Experiment
from experiments.utils.experiment_runner import run_multiple_pipelines
from modyn.config.schema.pipeline import (
    EvalHandlerConfig,
    ModynPipelineConfig,
)
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerExecutionTime
from modyn.config.schema.pipeline.evaluation.strategy.between_two_triggers import (
    BetweenTwoTriggersEvalStrategyConfig,
)
from modyn.config.schema.pipeline.evaluation.strategy.periodic import (
    PeriodicEvalStrategyConfig,
)
from modyn.config.schema.pipeline.evaluation.strategy.slicing import (
    SlicingEvalStrategyConfig,
)
from modyn.config.schema.pipeline.trigger.simple.data_amount import DataAmountTriggerConfig
from modyn.config.schema.pipeline.trigger.simple.time import TimeTriggerConfig
from modyn.utils.utils import SECONDS_PER_UNIT
from modynclient.config.schema.client_config import ModynClientConfig, Supervisor

_FIRST_TIMESTAMP = int(pd.to_datetime("1995-01-01").timestamp())
_LAST_TIMESTAMP = int(pd.to_datetime("2024-07-01").timestamp())

def construct_slicing_eval_handler(
    execution_time: EvalHandlerExecutionTime = "manual"
) -> EvalHandlerConfig:
    return EvalHandlerConfig(
        name="slidingmatrix",
        execution_time=execution_time,
        models="matrix",
        strategy=SlicingEvalStrategyConfig(
            eval_every="26w",
            eval_start_from=_FIRST_TIMESTAMP,
            eval_end_at=_LAST_TIMESTAMP,
        ),
        datasets=["arxiv_kaggle_test"],
    )


def construct_periodic_eval_handlers(
    intervals: list[tuple[str, str]], execution_time: EvalHandlerExecutionTime = "manual"
) -> list[EvalHandlerConfig]:
    """
    Args:
        intervals: List of (handler_name_suffix, interval string expression)
    """
    return [
        EvalHandlerConfig(
            name=f"periodic-{interval}",
            execution_time=execution_time,
            models="matrix",
            strategy=PeriodicEvalStrategyConfig(
                every="26w",
                interval=f"[-{fake_interval}; +{fake_interval}]",
                start_timestamp=_FIRST_TIMESTAMP + 13 * SECONDS_PER_UNIT["w"],
                end_timestamp=_LAST_TIMESTAMP,
            ),
            datasets=["arxiv_kaggle_test"],
        )
        for (interval, fake_interval) in intervals
    ]


def construct_between_trigger_eval_handler(execution_time: EvalHandlerExecutionTime = "manual") -> EvalHandlerConfig:
    return EvalHandlerConfig(
        name="full",
        execution_time=execution_time,
        models="active",
        strategy=BetweenTwoTriggersEvalStrategyConfig(),
        datasets=["arxiv_kaggle_all"],  # train and test
    )


def construct_pipelines(experiment: Experiment) -> list[ModynPipelineConfig]:
    return [
        gen_pipeline_config(
            config_ref=f"{trigger_name}",
            trigger_config=trigger_config,
            eval_handlers=experiment.eval_handlers,
            gpu_device=experiment.gpu_device,
            seed=experiment.seed,
        )
        for trigger_name, trigger_config in (
            [(f"timetrigger_{_name}", _conf) for _name, _conf in experiment.time_triggers.items()]
            + [(f"dataamount_{_name}", _conf) for _name, _conf in experiment.data_amount_triggers.items()]
            + [(f"drifttrigger_{_name}", _conf) for _name, _conf in experiment.drift_detection_triggers.items()]
            + [(f"performancetrigger_{_name}", _conf) for _name, _conf in experiment.performance_triggers.items()]
            + [(f"costtrigger_{_name}", _conf) for _name, _conf in experiment.cost_triggers.items()]
        )
    ]

PERIODIC_EVAL_INTERVAL = [("current", "13w")]  # total: 1/2y

# pretrain/cold start can be chosen post fuction by just dropping evaluation info before a certain date
_EXPERIMENT_REFS: dict[int, Experiment] = {
    # -------------------------------------------------------------------------------- #
    #         1X: Baselines with PERIODIC_EVAL_INTERVAL, executed with cautious        #
    #              parallelism and post factum evaluation (bottlenecking)              #
    # -------------------------------------------------------------------------------- #
    # TODO: merge main
    # TODO: reset datasets in db
    # time baselines
    10: Experiment(
        name="arxiv-baseline-time",
        eval_handlers=(
            construct_periodic_eval_handlers(intervals=PERIODIC_EVAL_INTERVAL, execution_time="manual") +
            construct_between_trigger_eval_handler("manual")
        ),
        time_triggers={
            schedule: TimeTriggerConfig(every=schedule, start_timestamp=_FIRST_TIMESTAMP)
            for schedule in reversed(["52w", "2y", "5y", "10y"])  # TODO: add 1y
        },
        gpu_device="cuda:0",
    ),
    # data amount baselines
    11: Experiment(
        name="arxiv-baseline-dataamount",
        eval_handlers=(
            construct_periodic_eval_handlers(intervals=PERIODIC_EVAL_INTERVAL, execution_time="manual") +
            construct_between_trigger_eval_handler("manual")
        ),
        data_amount_triggers={
            f"{num_samples}": DataAmountTriggerConfig(num_samples=num_samples)
            for num_samples in reversed([50_000, 100_000, 500_000])  # TODO: add 25_000
        },
        gpu_device="cuda:1",
    ),
    # -------------------------------------------------------------------------------- #
    #                                2X: Drift triggers                                #
    # -------------------------------------------------------------------------------- #
    # TODO
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
