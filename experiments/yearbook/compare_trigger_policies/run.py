import os

from experiments.utils.experiment_runner import run_multiple_pipelines
from experiments.utils.models import Experiment
from experiments.yearbook.compare_trigger_policies.pipeline_config import (
    gen_pipeline_config,
)
from modyn.config.schema.pipeline import (
    ModynPipelineConfig,
    TimeTriggerConfig,
)
from modyn.config.schema.pipeline.evaluation.config import EvalDataConfig
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig, EvalHandlerExecutionTime
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
from modyn.config.schema.pipeline.trigger.cost.cost import (
    AvoidableMisclassificationCostTriggerConfig,
    DataIncorporationLatencyCostTriggerConfig,
)
from modyn.config.schema.pipeline.trigger.performance.performance import (
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


def construct_slicing_eval_handler(execution_time: EvalHandlerExecutionTime = "manual") -> EvalHandlerConfig:
    return [
        EvalHandlerConfig(
            name="slidingmatrix",
            execution_time=execution_time,
            models="matrix",
            strategy=SlicingEvalStrategyConfig(
                eval_every="1d",
                eval_start_from=_FIRST_TIMESTAMP,
                eval_end_at=_LAST_TIMESTAMP,
            ),
            datasets=["yearbook_test"],
        )
    ]


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
                every="1d",  # every year
                interval=f"[-{fake_interval}; +{fake_interval}]",
                start_timestamp=_FIRST_TIMESTAMP,
                end_timestamp=_LAST_TIMESTAMP,
            ),
            datasets=["yearbook_test"],
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
            datasets=["yearbook_all"],  # train and test
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


# TODO: rerun with different seeds
_ALL_PERIODIC_EVAL_INTERVALS = [
    ("current", "20h"),  # total: 1 year
    ("delta+-1y", f"{1*24+3}h"),  # total: 3 years
    ("delta+-2y", f"{2*24+3}h"),  # total: 5 years
    ("delta+-3y", f"{3*24+3}h"),
    ("delta+-5y", f"{5*24+3}h"),
    ("delta+-10y", f"{10*24+3}h"),
    ("delta+-15y", f"{15*24+3}h"),
]

BEST_PERIODIC_EVAL_INTERVAL = [("delta+-1y", f"{1*24+3}h")]  # total: 3 years

_EXPERIMENT_REFS = {
    # 0: Experiment(
    #     name="yb-dev",
    #     eval_handlers=(
    #         construct_periodic_eval_handlers(intervals=[
    #             ("current", "20h"),  # total: 1 year
    #             ("delta+-1y", f"{1*24+3}h"),  # total: 3 years
    #         ], execution_time="after_pipeline")
    #         # construct_slicing_eval_handler("after_pipeline") +
    #         # construct_between_trigger_eval_handler("after_pipeline")
    #     ),
    #     time_triggers={
    #         "20y": TimeTriggerConfig(every="20d", start_timestamp=_FIRST_TIMESTAMP)
    #     },
    #     gpu_device="cuda:0",
    # ),
    # -------------------------------------------------------------------------------- #
    #             0X: Baselines with varying periodic evaluation intervals             #
    # -------------------------------------------------------------------------------- #
    # # time baselines
    # 1: Experiment(
    #     name="yb-baseline-time",
    #     eval_handlers=(
    #         construct_periodic_eval_handlers(intervals=_ALL_PERIODIC_EVAL_INTERVALS, execution_time="after_pipeline") +
    #         construct_between_trigger_eval_handler("after_pipeline")
    #     ),
    #     time_triggers={
    #         f"{schedule}y": TimeTriggerConfig(every=f"{schedule}d", start_timestamp=_FIRST_TIMESTAMP)
    #         for schedule in reversed([1, 2, 3, 4, 5, 10, 15, 25, 40])
    #     },
    #     gpu_device="cuda:3",
    # ),
    # # data amount baselines
    # 2: Experiment(
    #     name="yb-baseline-dataamount",
    #     eval_handlers=(
    #         construct_periodic_eval_handlers(intervals=_ALL_PERIODIC_EVAL_INTERVALS, execution_time="after_pipeline") +
    #         construct_between_trigger_eval_handler("after_pipeline")
    #     ),
    #     data_amount_triggers={
    #         f"{num_samples}": DataAmountTriggerConfig(num_samples=num_samples)
    #         for num_samples in reversed([250, 500, 1_000, 2_500, 5_000, 10_000, 15_000, 30_000])
    #     },
    #     gpu_device="cuda:2",
    # ),
    # -------------------------------------------------------------------------------- #
    #      1X: Baselines with BEST_PERIODIC_EVAL_INTERVAL, executed with cautious      #
    #              parallelism and post factum evaluation (bottlenecking)              #
    # -------------------------------------------------------------------------------- #
    # # time baselines
    # 10: Experiment(
    #     name="yb-baseline-time",
    #     eval_handlers=(
    #         construct_periodic_eval_handlers(intervals=BEST_PERIODIC_EVAL_INTERVAL, execution_time="manual") +
    #         construct_between_trigger_eval_handler("manual")
    #     ),
    #     time_triggers={
    #         f"{schedule}y": TimeTriggerConfig(every=f"{schedule}d", start_timestamp=_FIRST_TIMESTAMP)
    #         for schedule in reversed([1, 2, 3, 4, 5, 10, 15, 25, 40])
    #     },
    #     gpu_device="cuda:1",
    # ),
    # # data amount baselines
    # 11: Experiment(
    #     name="yb-baseline-dataamount",
    #     eval_handlers=(
    #         construct_periodic_eval_handlers(intervals=BEST_PERIODIC_EVAL_INTERVAL, execution_time="manual") +
    #         construct_between_trigger_eval_handler("manual")
    #     ),
    #     data_amount_triggers={
    #         f"{num_samples}": DataAmountTriggerConfig(num_samples=num_samples)
    #         for num_samples in ([250, 500, 1_000, 2_500, 5_000, 10_000, 15_000, 30_000])
    #     },
    #     gpu_device="cuda:2",
    # ),
    # -------------------------------------------------------------------------------- #
    #                                2X: Drift triggers                                #
    # -------------------------------------------------------------------------------- #
    # Static threshold drift
    # 20: Experiment(
    #     name="yb-datadrift-static",
    #     eval_handlers=(
    #         construct_periodic_eval_handlers(intervals=BEST_PERIODIC_EVAL_INTERVAL, execution_time="manual") +
    #         construct_between_trigger_eval_handler("manual")
    #     ),
    #     drift_detection_triggers={
    #         f"{criterion_name}_int{detection_interval}_win{window_size}": DataDriftTriggerConfig(
    #             evaluation_interval_data_points=detection_interval,
    #             windowing_strategy=TimeWindowingStrategy(
    #                 # overlap has no affect acc. to offline exploration
    #                 limit_ref=window_size, limit_cur=window_size, allow_overlap=False
    #             ),
    #             # with 30k samples and 84 years, 10y are roughly 30000/84*10=3500 samples
    #             # hence, if we want ~10 years of warmup, to 3500/detection_interval warmup intervals
    #             warmup_intervals=3500 // detection_interval,
    #             # triggering every 3 years during the warmup phase seems reasonable.
    #             warmup_policy=TimeTriggerConfig(every="3d", start_timestamp=_FIRST_TIMESTAMP),
    #             # 5k samples are enough for drift detection, in yearbook we won't accumulate that many anyway
    #             sample_size=5_000,
    #             metrics={
    #                 "mmd": AlibiDetectMmdDriftMetric(
    #                     decision_criterion=criterion, device="gpu"
    #                 )
    #             },
    #         )
    #         for detection_interval in [100, 250, 500, 1_000]
    #         for window_size in ["1d", "4d", "10d"]
    #         for threshold in reversed([0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.2, 0.4])
    #         # multiprocessing across gpus
    #         # 1: 0.4, 0.03, 0.09
    #         # 2: 0.2, 0.05, 0.07
    #         # 3: 0.15, 0.12
    #         # rerun failed
    #         # for threshold, detection_interval, window_size in [
    #         #     # (0.03, 250, "10d"),
    #         # ]
    #         for criterion_name, criterion in {
    #             f"mmd-{threshold}": ThresholdDecisionCriterion(threshold=threshold)
    #         }.items()
    #     },
    #     gpu_device="cuda:3",
    # ),
    # # Dynamic threshold drift
    # 21: Experiment(
    #     name="yb-datadrift-dynamic",
    #     eval_handlers=(
    #         construct_periodic_eval_handlers(intervals=BEST_PERIODIC_EVAL_INTERVAL, execution_time="manual")
    #         + construct_between_trigger_eval_handler("manual")
    #     ),
    #     drift_detection_triggers={
    #         f"{criterion_name}_int{detection_interval}_win{window_size}": DataDriftTriggerConfig(
    #             evaluation_interval_data_points=detection_interval,
    #             windowing_strategy=TimeWindowingStrategy(
    #                 # overlap has no affect acc. to offline exploration
    #                 limit_ref=window_size,
    #                 limit_cur=window_size,
    #                 allow_overlap=False,
    #             ),
    #             # with 30k samples and 84 years, 10y are roughly 30000/84*10=3500 samples
    #             # hence, if we want ~10 years of warmup, to 3500/detection_interval warmup intervals
    #             warmup_intervals=3500 // detection_interval,
    #             # triggering every 3 years during the warmup phase seems reasonable.
    #             warmup_policy=TimeTriggerConfig(every="3d", start_timestamp=_FIRST_TIMESTAMP),
    #             # 5k samples are enough for drift detection, in yearbook we won't accumulate that many anyway
    #             sample_size=5_000,
    #             metrics={"mmd": AlibiDetectMmdDriftMetric(decision_criterion=criterion, device="gpu")},
    #         )
    #         # multiprocessing across gpus
    #         for detection_interval in reversed([100, 250, 500])
    #         for window_size in ["4d"]  # dataset specific, best acc. to offline exploraion and static drift experiments
    #         for decision_window_size in [10, 20, 30]
    #         # cuda:1: 10
    #         # cuda:2: 20
    #         # cuda:3: 30
    #         for criterion_name, criterion in (
    #             {
    #                 f"mmd-quant-{quantile}-{decision_window_size}": DynamicQuantileThresholdCriterion(
    #                     window_size=decision_window_size, quantile=quantile
    #                 )
    #                 for quantile in [0.05, 0.1, 0.15, 0.3]
    #             }
    #             |
    #             {
    #                 f"mmd-rollavg-{deviation}-{decision_window_size}": DynamicRollingAverageThresholdCriterion(
    #                     window_size=decision_window_size, deviation=deviation, absolute=False
    #                 )
    #                 for deviation in [0.05, 0.2, 0.5, 1.0, 2.0]
    #             }
    #         ).items()
    #     },
    #     gpu_device="cuda:0",
    # ),
    # # -------------------------------------------------------------------------------- #
    # #                             3X:  Performance triggers                            #
    # # -------------------------------------------------------------------------------- #
    # 30: Experiment(
    #     name="yb-performancetrigger",
    #     eval_handlers=(
    #         construct_periodic_eval_handlers(intervals=BEST_PERIODIC_EVAL_INTERVAL, execution_time="manual")
    #         + construct_between_trigger_eval_handler("manual")
    #     ),
    #     performance_triggers={
    #         f"{criterion_name}-int{detection_interval}y": PerformanceTriggerConfig(
    #             evaluation_interval_data_points=detection_interval,
    #             data_density_window_size=20,  # performed well for drift, only used for #avoidable misclass
    #             performance_triggers_window_size=20,  # performed well for drift, only used for #avoidable misclass
    #             warmup_intervals=3500 // detection_interval,  # same as in drift case
    #             warmup_policy=TimeTriggerConfig(every="3d", start_timestamp=_FIRST_TIMESTAMP),
    #             evaluation=PerformanceTriggerEvaluationConfig(
    #                 device="cuda:2",
    #                 dataset=EvalDataConfig(
    #                     dataset_id="yearbook_train",  # optional: extra holdout split
    #                     bytes_parser_function=yb_bytes_parser_function,
    #                     batch_size=512,
    #                     dataloader_workers=1,
    #                     metrics=[
    #                         AccuracyMetricConfig(evaluation_transformer_function=yb_evaluation_transformer_function),
    #                     ],
    #                 ),
    #             ),
    #             mode="hindsight",
    #             forecasting_method="ridge_regression",
    #             decision_criteria={criterion_name: criterion},
    #         )
    #         # for detection_interval in [100, 250, 500]
    #         for detection_interval in [250]  # Solid choice
    #         for criterion_name, criterion in (
    #             {
    #                 f"static-{perf_threshold}": StaticPerformanceThresholdCriterion(
    #                     metric="Accuracy", metric_threshold=perf_threshold
    #                 )
    #                 for perf_threshold in [0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.925, 0.95]
    #             }
    #             | {
    #                 f"dynamic-quant-{quantile}-{decision_window_size}": DynamicQuantilePerformanceThresholdCriterion(
    #                     metric="Accuracy",
    #                     quantile=quantile,
    #                     window_size=decision_window_size,
    #                 )
    #                 for quantile in [0.05, 0.15, 0.3]
    #                 for decision_window_size in [10, 20, 30]
    #             }
    #             |
    #             {   # only executed for 250 and 500 detection intervals
    #                 f"dynamic-rollavg-{deviation}-{decision_window_size}": DynamicRollingAveragePerformanceThresholdCriterion(
    #                     metric="Accuracy",
    #                     deviation=deviation,
    #                     absolute=False,
    #                     window_size=decision_window_size,
    #                 )
    #                 for deviation in reversed([0.05, 0.1, 0.2, 0.3])
    #                 for decision_window_size in [10, 20, 30]
    #             }
    #             |
    #             {
    #                 # only executed for 250 detection interval
    #                 f"num_misclass-{num_misclassifications}-exp-{expected_accuracy}-red-{allow_reduction}-": StaticNumberAvoidableMisclassificationCriterion(
    #                     expected_accuracy=expected_accuracy,
    #                     allow_reduction=allow_reduction,
    #                     avoidable_misclassification_threshold=num_misclassifications,
    #                 )
    #                 # for num_misclassifications, expected_accuracy, allow_reduction in [
    #                 #     (1500, 0.95, False),
    #                 # ]
    #                 for num_misclassifications in reversed([50, 100, 200, 500, 1000, 1500])
    #                 for expected_accuracy in [0.85, 0.9, 0.95]
    #                 for allow_reduction in [True, False]
    #             }
    #         ).items()
    #     },
    #     gpu_device="cuda:2",
    # ),
    # -------------------------------------------------------------------------------- #
    #                              4X: Cost aware triggers                             #
    # -------------------------------------------------------------------------------- #
    # Data integration latency trigger
    40: Experiment(
        name="yb-costtrigger-dataincorporation",
        eval_handlers=(
            construct_periodic_eval_handlers(intervals=BEST_PERIODIC_EVAL_INTERVAL, execution_time="manual")
            + construct_between_trigger_eval_handler("manual")
        ),
        cost_triggers={
            f"int{interval}_exch{exchange_rate}": DataIncorporationLatencyCostTriggerConfig(
                evaluation_interval_data_points=interval,
                cost_tracking_window_size=20,
                incorporation_delay_per_training_second=exchange_rate,
            )
            for interval in reversed([100, 250, 500, 1_000])
            for exchange_rate in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        },
        gpu_device="cuda:0",
    ),
    # avoidable misclassfication integration trigger
    41: Experiment(
        name="yb-costtrigger-avoidablemisclassification",
        eval_handlers=(
            construct_periodic_eval_handlers(intervals=BEST_PERIODIC_EVAL_INTERVAL, execution_time="manual")
            + construct_between_trigger_eval_handler("manual")
        ),
        cost_triggers={
            f"int{detection_interval}_exch{exchange_rate}_red{allow_reduction}": AvoidableMisclassificationCostTriggerConfig(
                # cost trigger params
                expected_accuracy=0.9,  # assumed to work out ask it worked well for performance triggers
                cost_tracking_window_size=50,
                avoidable_misclassification_latency_per_training_second=exchange_rate,
                # performance trigger params
                evaluation_interval_data_points=detection_interval,
                data_density_window_size=20,
                performance_triggers_window_size=20,
                warmup_intervals=3500 // detection_interval,  # same as in drift case
                warmup_policy=TimeTriggerConfig(every="3d", start_timestamp=_FIRST_TIMESTAMP),
                evaluation=PerformanceTriggerEvaluationConfig(
                    device="cuda:1",
                    dataset=EvalDataConfig(
                        dataset_id="yearbook_train",
                        bytes_parser_function=yb_bytes_parser_function,
                        batch_size=512,
                        dataloader_workers=1,
                        metrics=[
                            AccuracyMetricConfig(evaluation_transformer_function=yb_evaluation_transformer_function),
                        ],
                    ),
                ),
                mode="hindsight",
                forecasting_method="ridge_regression",
            )
            # for detection_interval in [100, 250, 500]
            for detection_interval in [500]
            # cuda:1 - 100
            # cuda:2 - 250
            # cuda:3 - 500
            for exchange_rate in [1_000_000_000]
            for allow_reduction in [True, False]
        },
        gpu_device="cuda:1",
    ),
    # -------------------------------------------------------------------------------- #
    #                               5X: Ensemble triggers                              #
    # -------------------------------------------------------------------------------- #
    # # with best working previous triggers
    # 51: Experiment(
    #     name="yb-ensemble",
    #     eval_handlers=(
    #         construct_periodic_eval_handlers(intervals=BEST_PERIODIC_EVAL_INTERVAL, execution_time="manual")
    #         + construct_between_trigger_eval_handler("manual")
    #     ),
    #     ensemble_triggers={
    #         "ensemble1": EnsembleTriggerConfig(
    #             subtriggers={
    #                 "drift1": DataDriftTriggerConfig(
    #                     evaluation_interval_data_points=500,
    #                     windowing_strategy=TimeWindowingStrategy(limit_ref="4d", limit_cur="4d"),
    #                     warmup_intervals=10,
    #                     warmup_policy=TimeTriggerConfig(every="3d", start_timestamp=_FIRST_TIMESTAMP),
    #                     metrics={
    #                         "mmd": AlibiDetectMmdDriftMetric(
    #                             device="gpu",
    #                             decision_criterion=DynamicRollingAverageThresholdCriterion(
    #                                 deviation=0.1, absolute=False, window_size=15
    #                             ),
    #                         )
    #                     },
    #                 ),
    #                 "perf1": PerformanceTriggerConfig(
    #                     evaluation_interval_data_points=500,
    #                     data_density_window_size=20,
    #                     performance_triggers_window_size=20,
    #                     evaluation=PerformanceTriggerEvaluationConfig(
    #                         device="cuda:0",
    #                         dataset=EvalDataConfig(
    #                             dataset_id="yearbook_train",
    #                             bytes_parser_function=yb_bytes_parser_function,
    #                             batch_size=64,
    #                             dataloader_workers=1,
    #                             metrics=[
    #                                 AccuracyMetricConfig(
    #                                     evaluation_transformer_function=yb_evaluation_transformer_function
    #                                 ),
    #                             ],
    #                         ),
    #                     ),
    #                     mode="hindsight",  # TODO: lookahead
    #                     forecasting_method="ridge_regression",
    #                     decision_criteria={
    #                         "static-0.8": StaticPerformanceThresholdCriterion(metric="Accuracy", metric_threshold=0.8)
    #                     },
    #                 ),
    #             },
    #             ensemble_strategy=AtLeastNEnsembleStrategy(n=1),
    #         )
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
