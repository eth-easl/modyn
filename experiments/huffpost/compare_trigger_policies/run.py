import os

import pandas as pd

from experiments.utils.experiment_runner import run_multiple_pipelines
from experiments.utils.models import Experiment
from modyn.config.schema.pipeline import (
    EvalHandlerConfig,
    ModynPipelineConfig,
)
from modyn.config.schema.pipeline.evaluation.config import EvalDataConfig
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerExecutionTime
from modyn.config.schema.pipeline.evaluation.metrics import AccuracyMetricConfig
from modyn.config.schema.pipeline.evaluation.strategy.between_two_triggers import (
    BetweenTwoTriggersEvalStrategyConfig,
)
from modyn.config.schema.pipeline.evaluation.strategy.periodic import (
    PeriodicEvalStrategyConfig,
)
from modyn.config.schema.pipeline.evaluation.strategy.slicing import (
    SlicingEvalStrategyConfig,
)
from modyn.config.schema.pipeline.trigger.drift.alibi_detect import AlibiDetectMmdDriftMetric
from modyn.config.schema.pipeline.trigger.drift.config import DataDriftTriggerConfig
from modyn.config.schema.pipeline.trigger.drift.criterion import (
    DynamicQuantileThresholdCriterion,
    DynamicRollingAverageThresholdCriterion,
)
from modyn.config.schema.pipeline.trigger.drift.detection_window.time_ import TimeWindowingStrategy
from modyn.config.schema.pipeline.trigger.performance.criterion import (
    StaticNumberAvoidableMisclassificationCriterion,
)
from modyn.config.schema.pipeline.trigger.performance.performance import (
    PerformanceTriggerConfig,
    PerformanceTriggerEvaluationConfig,
)
from modyn.config.schema.pipeline.trigger.simple.data_amount import DataAmountTriggerConfig
from modyn.config.schema.pipeline.trigger.simple.time import TimeTriggerConfig
from modynclient.config.schema.client_config import ModynClientConfig, Supervisor

from .pipeline_config import (
    gen_pipeline_config,
    hp_bytes_parser_function,
    hp_evaluation_transformer_function,
)

_FIRST_TIMESTAMP = int(pd.to_datetime("2012-01-28").timestamp())
_LAST_TIMESTAMP = int(pd.to_datetime("2022-09-24").timestamp())  # last: dummy


def construct_slicing_eval_handler(execution_time: EvalHandlerExecutionTime = "manual") -> EvalHandlerConfig:
    return EvalHandlerConfig(
        name="slidingmatrix",
        execution_time=execution_time,
        models="matrix",
        strategy=SlicingEvalStrategyConfig(
            eval_every="13w",  # once per quarter
            eval_start_from=_FIRST_TIMESTAMP,
            eval_end_at=_LAST_TIMESTAMP,
        ),
        datasets=["huffpost_kaggle_test"],
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
                every="13w",  # once per quarter
                interval=f"[-{fake_interval}; +{fake_interval}]",
                start_timestamp=_FIRST_TIMESTAMP,
                end_timestamp=_LAST_TIMESTAMP,
            ),
            datasets=["huffpost_kaggle_test"],
        )
        for (interval, fake_interval) in intervals
    ]


def construct_between_trigger_eval_handler(
    execution_time: EvalHandlerExecutionTime = "manual",
) -> list[EvalHandlerConfig]:
    return [
        EvalHandlerConfig(
            name="full",
            execution_time=execution_time,
            models="active",
            strategy=BetweenTwoTriggersEvalStrategyConfig(),
            datasets=["huffpost_kaggle_all"],  # train and test
        )
    ]


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


# total: 14weeks -> ~4mths (with quarterly evaluations the intervals slightly overlap by 1 month)
PERIODIC_EVAL_INTERVAL = [("current", "7w")]

_EXPERIMENT_REFS: dict[int, Experiment] = {
    # -------------------------------------------------------------------------------- #
    #         1X: Baselines with PERIODIC_EVAL_INTERVAL, executed with cautious        #
    #              parallelism and post factum evaluation (bottlenecking)              #
    # -------------------------------------------------------------------------------- #
    # # time baselines
    # 10: Experiment(
    #     name="hp-baseline-time",
    #     eval_handlers=(
    #         construct_periodic_eval_handlers(intervals=PERIODIC_EVAL_INTERVAL, execution_time="manual")
    #         + construct_between_trigger_eval_handler("manual")
    #     ),
    #     time_triggers={
    #         schedule: TimeTriggerConfig(every=schedule, start_timestamp=_FIRST_TIMESTAMP)
    #         for schedule in (["13w", "4y"])  # reversed
    #         # 0: "26w", "1y", "2y"
    #         # 1: "13w", "4y"
    #     },
    #     gpu_device="cuda:2",
    # ),
    # # data amount baselines
    11: Experiment(
        name="hp-baseline-dataamount",
        eval_handlers=(
            construct_periodic_eval_handlers(intervals=PERIODIC_EVAL_INTERVAL, execution_time="manual")
            + construct_between_trigger_eval_handler("manual")
        ),
        data_amount_triggers={
            f"{num_samples}": DataAmountTriggerConfig(num_samples=num_samples)
            for num_samples in ([15_000, 30_000]) # 5_000, 10_000, 15_000, 20_000, 30_000, 40_000, 80_000
        },
        gpu_device="cuda:2",
    ),
    # -------------------------------------------------------------------------------- #
    #                                2X: Drift triggers                                #
    # -------------------------------------------------------------------------------- #
    21: Experiment(
        name="hp-datadrift-dynamic",
        eval_handlers=(
            construct_periodic_eval_handlers(intervals=PERIODIC_EVAL_INTERVAL, execution_time="manual")
            # + construct_between_trigger_eval_handler("manual")  # not executed to speed things up
        ),
        drift_detection_triggers={
            f"{criterion_name}_int{detection_interval}_win{window_size}": DataDriftTriggerConfig(
                evaluation_interval_data_points=detection_interval,
                windowing_strategy=TimeWindowingStrategy(
                    # overlap has no affect acc. to offline exploration
                    limit_ref=window_size,
                    limit_cur=window_size,
                    allow_overlap=False,
                ),
                # first 200k of 2mio samples are warmup
                warmup_intervals=30_000 // detection_interval,
                # triggering every 3 years during the warmup phase seems reasonable.
                warmup_policy=TimeTriggerConfig(every="40w", start_timestamp=_FIRST_TIMESTAMP),
                # 5k samples are enough for drift detection
                sample_size=5_000,
                metrics={"mmd": AlibiDetectMmdDriftMetric(decision_criterion=criterion, device="gpu")},
            )
            # multiprocessing across gpus
            for detection_interval in [1500]
            for window_size in ["1y"]  # dataset specific
            for decision_window_size in [20]  # more values
            for criterion_name, criterion in (
                {
                    f"mmd-quant-{quantile}-{decision_window_size}": DynamicQuantileThresholdCriterion(
                        window_size=decision_window_size, quantile=quantile
                    )
                    for quantile in [0.05, 0.10, 0.15]  # TODO: 0.3
                    # cuda3
                }
                |
                {
                    f"mmd-rollavg-{deviation}-{decision_window_size}": DynamicRollingAverageThresholdCriterion(
                        window_size=decision_window_size, deviation=deviation, absolute=False
                    )
                    for deviation in reversed([0.5, 1.0, 2.0, 5.0])  # TODO: 0.05, 0.2,
                    # cuda3
                }
            ).items()
        },
        gpu_device="cuda:3",
    ),
    # -------------------------------------------------------------------------------- #
    #                             3X:  Performance triggers                            #
    # -------------------------------------------------------------------------------- #
    30: Experiment(
        name="hp-performancetrigger",
        eval_handlers=(
            construct_periodic_eval_handlers(intervals=PERIODIC_EVAL_INTERVAL, execution_time="manual")
            + construct_between_trigger_eval_handler("manual")
        ),
        performance_triggers={
            f"{criterion_name}-int{detection_interval}y": PerformanceTriggerConfig(
                evaluation_interval_data_points=detection_interval,
                data_density_window_size=20,  # performed well for drift, only used for #avoidable misclass
                performance_triggers_window_size=20,  # performed well for drift, only used for #avoidable misclass
                warmup_intervals=30_000 // detection_interval,  # first 200k of 2mio samples are warmup
                # triggering every 3 years during the warmup phase seems reasonable.
                warmup_policy=TimeTriggerConfig(every="40w", start_timestamp=_FIRST_TIMESTAMP),
                evaluation=PerformanceTriggerEvaluationConfig(
                    device="cuda:2",
                    dataset=EvalDataConfig(
                        dataset_id="huffpost_kaggle_train",  # optional: extra holdout split
                        bytes_parser_function=hp_bytes_parser_function,
                        batch_size=512,
                        dataloader_workers=1,
                        metrics=[
                            AccuracyMetricConfig(evaluation_transformer_function=hp_evaluation_transformer_function),
                        ],
                    ),
                ),
                mode="hindsight",
                forecasting_method="ridge_regression",
                decision_criteria={criterion_name: criterion},
            )
            for detection_interval in [1500]
            for criterion_name, criterion in (
                # {
                #     f"static-{perf_threshold}": StaticPerformanceThresholdCriterion(
                #         metric="Accuracy", metric_threshold=perf_threshold
                #     )
                #     for perf_threshold in [0.45, 0.5, 0.55, 0.6]
                # }
                # |
                # {
                #     f"dynamic-quant-{quantile}-{decision_window_size}": DynamicQuantilePerformanceThresholdCriterion(
                #         metric="Accuracy",
                #         quantile=quantile,
                #         window_size=decision_window_size,
                #     )
                #     for quantile in [0.05, 0.15, 0.3]
                #     for decision_window_size in [15, 30]
                # }
                # |
                # {
                #     f"dynamic-rollavg-{deviation}-{decision_window_size}": DynamicRollingAveragePerformanceThresholdCriterion(
                #         metric="Accuracy",
                #         deviation=deviation,
                #         absolute=False,
                #         window_size=decision_window_size,
                #     )
                #     for deviation in reversed([0.1, 0.2, 0.3])
                #     for decision_window_size in [15, 30]
                # }
                # |
                {
                    f"num_misclass-{num_misclassifications}-exp-{expected_accuracy}-red-{allow_reduction}-": StaticNumberAvoidableMisclassificationCriterion(
                        expected_accuracy=expected_accuracy,
                        allow_reduction=allow_reduction,
                        avoidable_misclassification_threshold=num_misclassifications,
                    )
                    # for num_misclassifications in reversed([250, 500, 1000, 4000, 8000])  # 250, 500, 1000, 4000
                    # for expected_accuracy in [0.5, 0.6]
                    # for allow_reduction in [False]
                    for num_misclassifications, expected_accuracy, allow_reduction in [
                        (500, 0.5, False),  # TODO:
                        (500, 0.6, False),
                        (250, 0.5, False),
                        (250, 0.6, False),
                    ]
                }
            ).items()
        },
        gpu_device="cuda:2",
    ),
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
