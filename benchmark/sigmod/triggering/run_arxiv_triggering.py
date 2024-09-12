from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

from benchmark.sigmod.triggering.arxiv_triggering_config import gen_arxiv_triggering_config, get_eval_data_config
from experiments.utils.experiment_runner import run_multiple_pipelines
from modyn.config.schema.pipeline import ModynPipelineConfig
from modyn.config.schema.pipeline.trigger import TriggerConfig
from modyn.config.schema.pipeline.trigger.drift import DataDriftTriggerConfig
from modyn.config.schema.pipeline.trigger.drift.alibi_detect import AlibiDetectMmdDriftMetric
from modyn.config.schema.pipeline.trigger.drift.criterion import (
    DynamicQuantileThresholdCriterion,
    DynamicRollingAverageThresholdCriterion,
    ThresholdDecisionCriterion,
)
from modyn.config.schema.pipeline.trigger.drift.detection_window.time_ import TimeWindowingStrategy
from modyn.config.schema.pipeline.trigger.performance.criterion import (
    DynamicQuantilePerformanceThresholdCriterion,
    DynamicRollingAveragePerformanceThresholdCriterion,
    StaticPerformanceThresholdCriterion,
)
from modyn.config.schema.pipeline.trigger.performance.performance import (
    PerformanceTriggerConfig,
    PerformanceTriggerEvaluationConfig,
)
from modyn.config.schema.pipeline.trigger.simple.data_amount import DataAmountTriggerConfig
from modyn.config.schema.pipeline.trigger.simple.time import TimeTriggerConfig
from modyn.supervisor.internal.pipeline_executor.models import PipelineLogs
from modyn.utils.utils import current_time_millis
from modynclient.config.schema.client_config import ModynClientConfig, Supervisor

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"client_{current_time_millis()}.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

START_TIMESTAMP = 631152000


def gen_triggering_strategies(device: str) -> list[tuple[str, TriggerConfig]]:
    strategies = []

    # TimeTriggers
    for years in [1, 3, 5]:
        strategies.append(
            (f"timetrigger_{years}y", TimeTriggerConfig(every=f"{years}y", start_timestamp=START_TIMESTAMP))
        )

    # DataAmountTriggers
    for count in [30000, 50000]:
        strategies.append((f"amounttrigger_{count}", DataAmountTriggerConfig(num_samples=count)))

    return strategies


def gen_revision_triggering_strategies(device: str) -> list[tuple[str, TriggerConfig]]:
    strategies = []
    min_warmup_data_points = 20000

    for evaluation_interval_data_points in [5000]:
        warmup_intervals = math.ceil(min_warmup_data_points / evaluation_interval_data_points)

        # Static Drift Triggers
        for threshold in [0.02, 0.01, 0.005, 0.002]:
            for window_size in ["1y", "2y"]:
                conf = DataDriftTriggerConfig(
                    evaluation_interval_data_points=evaluation_interval_data_points,
                    windowing_strategy=TimeWindowingStrategy(
                        allow_overlap=True, limit_ref=window_size, limit_cur=window_size
                    ),
                    sample_size=10000,
                    metrics={
                        "mmd_alibi": AlibiDetectMmdDriftMetric(
                            device="gpu",
                            num_permutations=None,
                            decision_criterion=ThresholdDecisionCriterion(threshold=threshold),
                        )
                    },
                    warmup_policy=TimeTriggerConfig(every="1y", start_timestamp=START_TIMESTAMP),
                    warmup_intervals=warmup_intervals,
                )
                name = f"mmdalibi_{evaluation_interval_data_points}_{threshold}_{window_size}"
                strategies.append((name, conf))

        ## Dynamic Drift Triggers
        for window_size in ["1y", "2y"]:
            for metric_window_size in [15, 30]:  # how many drift scores we use for calibrating the dynamic policy
                criteria = []
                for deviation in [1]:
                    criteria.append(
                        (
                            f"roll_{deviation}",
                            DynamicRollingAverageThresholdCriterion(
                                window_size=metric_window_size, deviation=deviation, absolute=False
                            ),
                        )
                    )
                for quantile in [0.05, 0.1, 0.2]:
                    criteria.append(
                        (
                            f"qt_{quantile}",
                            DynamicQuantileThresholdCriterion(window_size=metric_window_size, quantile=quantile),
                        )
                    )
                for dec_crit_str, decision_criterion in criteria:
                    conf = DataDriftTriggerConfig(
                        evaluation_interval_data_points=evaluation_interval_data_points,
                        windowing_strategy=TimeWindowingStrategy(
                            allow_overlap=True, limit_ref=window_size, limit_cur=window_size
                        ),
                        sample_size=10000,
                        metrics={
                            "mmd_alibi": AlibiDetectMmdDriftMetric(
                                device="gpu",
                                num_permutations=None,
                                decision_criterion=decision_criterion,
                            )
                        },
                        warmup_policy=TimeTriggerConfig(every="1y", start_timestamp=START_TIMESTAMP),
                        warmup_intervals=warmup_intervals,
                    )

                    name = f"mmdalibi_dyn_{evaluation_interval_data_points}_{metric_window_size}_{dec_crit_str}_{window_size}"
                    strategies.append((name, conf))

        ## Static PerformanceTriggers
        for threshold in [0.8, 0.75, 0.7]:
            conf = PerformanceTriggerConfig(
                evaluation_interval_data_points=evaluation_interval_data_points,
                performance_triggers_window_size=1,  # deprecated
                data_density_window_size=1,  # deprecated
                mode="hindsight",
                evaluation=PerformanceTriggerEvaluationConfig(
                    device=device, dataset=get_eval_data_config("arxiv_kaggle_train")
                ),
                decision_criteria={
                    f"static-{threshold}": StaticPerformanceThresholdCriterion(
                        metric="Top-2-Accuracy", metric_threshold=threshold
                    )
                },
            )
            name = f"perf_{evaluation_interval_data_points}_{threshold}"
            strategies.append((name, conf))

        ## Dynamic PerformanceTriggers
        for performance_triggers_window_size in [15, 30]:
            criteria = []
            for deviation in [0.2]:
                criterion = DynamicRollingAveragePerformanceThresholdCriterion(
                    metric="Top-2-Accuracy",
                    window_size=performance_triggers_window_size,
                    deviation=deviation,
                    absolute=False,
                )
                criteria.append((f"{performance_triggers_window_size}_roll_{deviation}", criterion))

            for quantile in [0.05, 0.1, 0.2]:
                criterion = DynamicQuantilePerformanceThresholdCriterion(
                    metric="Top-2-Accuracy", window_size=performance_triggers_window_size, quantile=quantile
                )
                criteria.append((f"{performance_triggers_window_size}_perc_{quantile}", criterion))

            for dec_crit_str, decision_criterion in criteria:
                conf = PerformanceTriggerConfig(
                    evaluation_interval_data_points=evaluation_interval_data_points,
                    performance_triggers_window_size=performance_triggers_window_size,
                    mode="hindsight",
                    evaluation=PerformanceTriggerEvaluationConfig(
                        device=device, dataset=get_eval_data_config("arxiv_kaggle_train")
                    ),
                    decision_criteria={f"dynamic-{dec_crit_str}": decision_criterion},
                    warmup_policy=TimeTriggerConfig(every="1y", start_timestamp=START_TIMESTAMP),
                    warmup_intervals=warmup_intervals,
                )
                name = f"perf_dyn_{evaluation_interval_data_points}_{dec_crit_str}"
                strategies.append((name, conf))

    return strategies


def run_experiment() -> None:
    logger.info("Grüeziwohl!")
    pipeline_configs: list[ModynPipelineConfig] = []
    train_gpu = "cuda:0"
    num_gpus = 1  # to parallelize across gpus
    gpu_id = 0
    seeds = [42, 99, 12]  # set to [None] to disable, should be 0-100
    skip_existing = True

    existing_pipelines = []
    if skip_existing:
        log_directory = Path(input("Please enter the directory in which to search for existing pipelines: ")) or Path(
            "/raid/modyn/maxi/sigmod/logs"
        )
        if not log_directory.exists():
            raise RuntimeError(f"{log_directory} does not exist.")

        names = list(log_directory.glob("**/.name"))

        for name_file in names:
            name = name_file.read_text()
            pipeline_file = name_file.parent / "pipeline.log"

            if not pipeline_file.exists():
                logger.info(f"{name_file} exists, but {pipeline_file} does not")
                continue

            try:
                parsed_log = PipelineLogs.model_validate_json(pipeline_file.read_text())
            except:
                print(f"Skipping file {pipeline_file} due to invalid format")
                continue

            seed = parsed_log.config.pipeline.training.seed
            existing_pipelines.append((name, seed))

        logger.info(f"Found these existing pipelines: {existing_pipelines}")

    existing_pipelines = set(existing_pipelines)
    run_id = 0
    for seed in seeds:
        for triggering_strategy_id, triggering_strategy in gen_triggering_strategies(
            train_gpu
        ) + gen_revision_triggering_strategies(train_gpu):
            if (
                isinstance(triggering_strategy, DataDriftTriggerConfig)
                or isinstance(triggering_strategy, PerformanceTriggerConfig)
            ) and seed != seeds[0]:
                continue  # only execute drift triggers once

            pipeline_config = gen_arxiv_triggering_config(
                triggering_strategy_id, train_gpu, triggering_strategy, seed, START_TIMESTAMP
            )

            if run_id % num_gpus == gpu_id and (pipeline_config.pipeline.name, seed) not in existing_pipelines:
                logger.info(f"Running {triggering_strategy_id} with seed {seed} on this GPU.")
                pipeline_configs.append(pipeline_config)

            run_id += 1

    print(f"Running {len(pipeline_configs)} pipelines in total now.")
    host = os.getenv("MODYN_SUPERVISOR_HOST")
    port = os.getenv("MODYN_SUPERVISOR_PORT")

    if not host:
        host = input("Enter the supervisors host address: ") or "localhost"
    if not port:
        port = int(input("Enter the supervisors port: ") or "3000")

    run_multiple_pipelines(
        client_config=ModynClientConfig(supervisor=Supervisor(ip=host, port=port)),
        pipeline_configs=pipeline_configs,
        start_replay_at=START_TIMESTAMP,
        stop_replay_at=None,
        maximum_triggers=None,
        show_eval_progress=False,
    )


if __name__ == "__main__":
    run_experiment()
