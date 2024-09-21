import os

from experiments.models import Experiment
from experiments.utils.experiment_runner import run_multiple_pipelines
from experiments.yearbook.compare_trigger_policies.pipeline_config import (
    gen_pipeline_config,
)
from modyn.config.schema.pipeline import (
    DataAmountTriggerConfig,
    ModynPipelineConfig,
    TimeTriggerConfig,
)
from modyn.config.schema.pipeline.evaluation.config import EvalDataConfig
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig
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
from modyn.config.schema.pipeline.trigger import DataDriftTriggerConfig
from modyn.config.schema.pipeline.trigger.cost.cost import (
    AvoidableMisclassificationCostTriggerConfig,
    DataIncorporationLatencyCostTriggerConfig,
)
from modyn.config.schema.pipeline.trigger.drift.alibi_detect import (
    AlibiDetectMmdDriftMetric,
)
from modyn.config.schema.pipeline.trigger.drift.criterion import (
    DynamicQuantileThresholdCriterion,
    DynamicRollingAverageThresholdCriterion,
    ThresholdDecisionCriterion,
)
from modyn.config.schema.pipeline.trigger.drift.detection_window.time_ import (
    TimeWindowingStrategy,
)
from modyn.config.schema.pipeline.trigger.ensemble import (
    AtLeastNEnsembleStrategy,
    EnsembleTriggerConfig,
)
from modyn.config.schema.pipeline.trigger.performance.criterion import (
    StaticNumberAvoidableMisclassificationCriterion,
    StaticPerformanceThresholdCriterion,
    _DynamicPerformanceThresholdCriterion,
)
from modyn.config.schema.pipeline.trigger.performance.performance import (
    PerformanceTriggerConfig,
    PerformanceTriggerEvaluationConfig,
)
from modyn.utils.utils import SECONDS_PER_UNIT
from modynclient.config.schema.client_config import ModynClientConfig, Supervisor

from .pipeline_config import (
    yb_bytes_parser_function,
    yb_evaluation_transformer_function,
)

_FIRST_TIMESTAMP = 0
_LAST_TIMESTAMP = SECONDS_PER_UNIT["d"] * (2014 - 1930)  # 2014: dummy


def construct_slicing_eval_handler() -> EvalHandlerConfig:
    return EvalHandlerConfig(
        name="slidingmatrix",
        execution_time="manual",
        models="matrix",
        strategy=SlicingEvalStrategyConfig(
            eval_every="1d",
            eval_start_from=_FIRST_TIMESTAMP,
            eval_end_at=_LAST_TIMESTAMP,
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
        execution_time="manual",
        models="active",
        strategy=BetweenTwoTriggersEvalStrategyConfig(),
        datasets=["yearbook_all"],  # train and test
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


# TODO: rerun with different seeds
_EXPERIMENT_REFS = {
    # ----------------------------------- Baselines ---------------------------------- #
    # time baselines
    1: Experiment(
        name="yb-baseline-time",
        eval_handlers=[
            construct_slicing_eval_handler(),
            construct_between_trigger_eval_handler(),
        ],
        time_triggers={
            f"{schedule}y": TimeTriggerConfig(every=f"{schedule}d", start_timestamp=_FIRST_TIMESTAMP)
            for schedule in [1, 2, 3, 5, 15, 25, 40]
        },
        gpu_device="cuda:0",
    ),
    # data amount baselines
    2: Experiment(
        name="yb-baseline-dataamount",
        eval_handlers=[
            construct_slicing_eval_handler(),
            construct_between_trigger_eval_handler(),
        ],
        data_amount_triggers={
            f"{num_samples}": DataAmountTriggerConfig(num_samples=num_samples)
            for num_samples in [
                100,
                200,
                500,
                1_000,
                2_500,
                5_000,
                10_000,
                15_000,
                30_000,
            ]
        },
        gpu_device="cuda:1",
    ),
    # -------------------------------- Drift triggers -------------------------------- #
    # Static threshold drift
    3: Experiment(
        name="yb-baseline-datadrift-static",
        eval_handlers=[
            construct_slicing_eval_handler(),
            construct_between_trigger_eval_handler(),
        ],
        drift_detection_triggers={
            f"{criterion_name}_int{interval}_win{window_size}": DataDriftTriggerConfig(
                evaluation_interval_data_points=interval,
                windowing_strategy=TimeWindowingStrategy(
                    limit_ref=f"{window_size}d",
                    limit_cur=f"{window_size}d",
                ),
                warmup_intervals=10,
                warmup_policy=TimeTriggerConfig(every="3d", start_timestamp=_FIRST_TIMESTAMP),
                metrics={
                    "mmd": AlibiDetectMmdDriftMetric(
                        decision_criterion=criterion,
                        device="cuda:2",
                    )
                },
            )
            for interval in [100, 250, 500, 1_000]
            for window_size in [1, 4, 10]
            for criterion_name, criterion in {
                f"mmd-{threshold}": ThresholdDecisionCriterion(threshold=threshold)
                for threshold in [0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.2]
            }.items()
        },
        gpu_device="cuda:2",
    ),
    # Dynamic threshold drift
    4: Experiment(
        name="yb-baseline-datadrift-dynamic",
        eval_handlers=[
            construct_slicing_eval_handler(),
            construct_between_trigger_eval_handler(),
        ],
        drift_detection_triggers={
            f"{criterion_name}_int{interval}_win{window_size}": DataDriftTriggerConfig(
                evaluation_interval_data_points=interval,
                windowing_strategy=TimeWindowingStrategy(
                    limit_ref=f"{window_size}d",
                    limit_cur=f"{window_size}d",
                ),
                warmup_intervals=10,
                warmup_policy=TimeTriggerConfig(every="3d", start_timestamp=_FIRST_TIMESTAMP),
                metrics={
                    "mmd": AlibiDetectMmdDriftMetric(
                        decision_criterion=criterion,
                        device="cuda:1",
                    )
                },
            )
            for interval in [250]  # TODO: [100, 250, 500, 1_000]
            for window_size in [5]
            for criterion_name, criterion in (
                {
                    f"mmd-perc-{quantile}-{window_size}": DynamicQuantileThresholdCriterion(
                        window_size=window_size, quantile=quantile
                    )
                    for quantile in [0.05, 0.1, 0.2, 0.3]
                    for window_size in [15]  # TODO [10, 20, 30]
                }
                | {
                    f"mmd-rollavg-{deviation}-{window_size}": DynamicRollingAverageThresholdCriterion(
                        window_size=window_size, deviation=deviation, absolute=False
                    )  # TODO: avg / quantile
                    for deviation in [0.025, 0.05, 0.1, 0.2, 0.3]
                    for window_size in [15]  # TODO [10, 20, 30]
                }
            ).items()
        },
        gpu_device="cuda:1",
    ),
    # ----------------------------- Performance triggers ----------------------------- #
    5: Experiment(
        name="yb-performancetrigger",
        eval_handlers=[
            construct_slicing_eval_handler(),
            construct_between_trigger_eval_handler(),
        ],
        performance_triggers={
            f"{criterion_name}-int{interval}y": PerformanceTriggerConfig(
                evaluation_interval_data_points=interval,
                data_density_window_size=20,
                performance_triggers_window_size=20,
                evaluation=PerformanceTriggerEvaluationConfig(
                    device="cuda:2",
                    dataset=EvalDataConfig(
                        dataset_id="yearbook_train",
                        bytes_parser_function=yb_bytes_parser_function,
                        batch_size=512,  # TODO: lower
                        dataloader_workers=1,
                        metrics=[
                            AccuracyMetricConfig(evaluation_transformer_function=yb_evaluation_transformer_function),
                        ],
                    ),
                ),
                mode="hindsight",  # TODO: lookahead
                forecasting_method="ridge_regression",
                decision_criteria={criterion_name: criterion},
            )
            for interval in [500]  # TODO: [100, 250, 500, 1_000]
            for criterion_name, criterion in (
                {
                    f"static-{perf_threshold}": StaticPerformanceThresholdCriterion(
                        metric="Accuracy", metric_threshold=perf_threshold
                    )
                    for perf_threshold in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
                }
                | {
                    f"dynamic-{deviation}": _DynamicPerformanceThresholdCriterion(
                        metric="Accuracy",
                        deviation=deviation,
                        absolute=False,
                    )
                    for deviation in [0.025, 0.05, 0.1, 0.2, 0.3]
                }
                | {
                    f"num_misclass-{num_misclassifications}-{allow_reduction}-": StaticNumberAvoidableMisclassificationCriterion(
                        expected_accuracy=0.9,  # TODO: variable
                        allow_reduction=allow_reduction,
                        avoidable_misclassification_threshold=num_misclassifications,
                    )  # TODO: avg / quantile
                    for num_misclassifications in [100, 200, 500, 1000, 2000, 5000]
                    for allow_reduction in [True, False]
                }
            ).items()
        },
        gpu_device="cuda:2",
    ),
    # TODO: add mixed performance trigger
    # ------------------------------ Cost aware triggers ----------------------------- #
    # Data integration latency trigger
    10: Experiment(
        name="yb-costtrigger-dataincorporation",
        eval_handlers=[
            construct_slicing_eval_handler(),
            construct_between_trigger_eval_handler(),
        ],
        cost_triggers={
            f"int{interval}_exch{exchange_rate}": DataIncorporationLatencyCostTriggerConfig(
                evaluation_interval_data_points=interval,
                cost_tracking_window_size=20,
                incorporation_delay_per_training_second=exchange_rate,
            )
            for interval in [100, 250, 500, 1_000]
            for exchange_rate in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        },
        gpu_device="cuda:3",
    ),
    # avoidable misclassfication integration trigger
    11: Experiment(
        name="yb-costtrigger-avoidablemisclassification",
        eval_handlers=[
            construct_slicing_eval_handler(),
            construct_between_trigger_eval_handler(),
        ],
        cost_triggers={
            f"int{interval}_exch{exchange_rate}_red{allow_reduction}": AvoidableMisclassificationCostTriggerConfig(
                # cost trigger params
                expected_accuracy=0.9,
                cost_tracking_window_size=50,
                avoidable_misclassification_latency_per_training_second=exchange_rate,
                # performance trigger params
                evaluation_interval_data_points=interval,
                data_density_window_size=20,
                performance_triggers_window_size=20,
                evaluation=PerformanceTriggerEvaluationConfig(
                    device="cuda:2",
                    dataset=EvalDataConfig(
                        dataset_id="yearbook_train",
                        bytes_parser_function=yb_bytes_parser_function,
                        batch_size=512,  # TODO: lower
                        dataloader_workers=1,
                        metrics=[
                            AccuracyMetricConfig(evaluation_transformer_function=yb_evaluation_transformer_function),
                        ],
                    ),
                ),
                mode="hindsight",  # TODO: lookahead
                forecasting_method="ridge_regression",
            )
            for interval in [100, 250, 500, 1_000]
            for exchange_rate in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
            for allow_reduction in [True, False]
        },
        gpu_device="cuda:1",
    ),
    # ------------------------------- Ensemble triggers ------------------------------
    # with best working previous triggers
    20: Experiment(
        name="yb-ensemble",
        eval_handlers=[
            construct_slicing_eval_handler(),
            construct_between_trigger_eval_handler(),
        ],
        ensemble_triggers={
            "ensemble1": EnsembleTriggerConfig(
                subtriggers={
                    "drift1": DataDriftTriggerConfig(
                        evaluation_interval_data_points=500,
                        windowing_strategy=TimeWindowingStrategy(limit_ref="4d", limit_cur="4d"),
                        warmup_intervals=10,
                        warmup_policy=TimeTriggerConfig(every="3d", start_timestamp=_FIRST_TIMESTAMP),
                        metrics={
                            "mmd": AlibiDetectMmdDriftMetric(
                                device="cuda:0",
                                decision_criterion=DynamicRollingAverageThresholdCriterion(
                                    deviation=0.1, absolute=False, window_size=15
                                ),
                            )
                        },
                    ),
                    "perf1": PerformanceTriggerConfig(
                        evaluation_interval_data_points=500,
                        data_density_window_size=20,
                        performance_triggers_window_size=20,
                        evaluation=PerformanceTriggerEvaluationConfig(
                            device="cuda:0",
                            dataset=EvalDataConfig(
                                dataset_id="yearbook_train",
                                bytes_parser_function=yb_bytes_parser_function,
                                batch_size=64,
                                dataloader_workers=1,
                                metrics=[
                                    AccuracyMetricConfig(
                                        evaluation_transformer_function=yb_evaluation_transformer_function
                                    ),
                                ],
                            ),
                        ),
                        mode="hindsight",  # TODO: lookahead
                        forecasting_method="ridge_regression",
                        decision_criteria={
                            "static-0.8": StaticPerformanceThresholdCriterion(metric="Accuracy", metric_threshold=0.8)
                        },
                    ),
                },
                ensemble_strategy=AtLeastNEnsembleStrategy(n=1),
            )
        },
        gpu_device="cuda:0",
    ),
    # ----------------------------- Evaluation intervals ----------------------------- #
    # 100: Experiment(
    #     name="yb-timetrigger1y-periodic-eval-intervals",
    #     eval_handlers=[construct_slicing_eval_handler()]
    #     + construct_periodic_eval_handlers(
    #         intervals=[
    #             ("-delta-current", "23h"),
    #             ("-delta+-1y", f"{1*24+1}h"),
    #             ("-delta+-2y", f"{2*24+1}h"),
    #             ("-delta+-3y", f"{3*24+1}h"),
    #             ("-delta+-5y", f"{5*24+1}h"),
    #             ("-delta+-10y", f"{10*24+1}h"),
    #             ("-delta+-15y", f"{15*24+1}h"),
    #         ]
    #     ),
    #     time_triggers={
    #         "1y": TimeTriggerConfig(every="1d", start_timestamp=_FIRST_TIMESTAMP)
    #     },
    #     data_amount_triggers={},
    #     drift_detection_triggers={},
    #     gpu_device="cuda:0",
    # ),
    # 30: Experiment(
    #     name="yb-drift-interval-cost",
    #     eval_handlers=[
    #         construct_slicing_eval_handler(),
    #         construct_between_trigger_eval_handler(),
    #     ],
    #     time_triggers={},
    #     data_amount_triggers={},
    #     drift_detection_triggers={
    #         f"detection_interval_{detection_interval}": DataDriftTriggerConfig(
    #             evaluation_interval_data_points=detection_interval,
    #             windowing_strategy=TimeWindowingStrategy(
    #                 limit_ref="4d", limit_cur="4d",
    #             ),
    #             warmup_intervals=10,
    #             warmup_policy=TimeTriggerConfig(
    #                 every="3d", start_timestamp=_FIRST_TIMESTAMP
    #             ),
    #             metrics={
    #                 "mmd": AlibiDetectMmdDriftMetric(
    #                     decision_criterion=DynamicThresholdCriterion(window_size=10),
    #                     device="cuda:0",
    #                 )
    #             }
    #         )
    #         for detection_interval in [100, 200, 500, 1_000, 2_500, 5_000, 10_000, 15_000]
    #     },
    #     gpu_device="cuda:0",
    # ),
}


def run_experiment() -> None:
    host = os.getenv("MODYN_SUPERVISOR_HOST")
    port_str = os.getenv("MODYN_SUPERVISOR_PORT")

    if not host:
        host = input("Enter the supervisors host address: ") or "localhost"
    if port_str:
        port = int(port_str)
    else:
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
