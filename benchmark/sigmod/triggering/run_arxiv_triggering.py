from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from benchmark.sigmod.triggering.arxiv_triggering_config import gen_arxiv_triggering_config
from experiments.utils.experiment_runner import run_multiple_pipelines
from modyn.config.schema.pipeline import ModynPipelineConfig
from modyn.config.schema.pipeline.trigger import TriggerConfig
from modyn.config.schema.pipeline.trigger.drift import DataDriftTriggerConfig
from modyn.config.schema.pipeline.trigger.performance.performance import (
    PerformanceTriggerConfig,
)
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


def gen_triggering_strategies(device: str) -> list[tuple[str, TriggerConfig]]:
    strategies = []

    # TimeTriggers
    for years in [1, 3, 5]:
        strategies.append((f"timetrigger_{years}y", TimeTriggerConfig(every=f"{years}y", start_timestamp=631152000)))

    # DataAmountTriggers
    # for count in [500, 1000, 2000, 10000]:
    #    strategies.append((f"amounttrigger_{count}", DataAmountTriggerConfig(num_samples=count)))

    # DriftTriggers
    # for evaluation_interval_data_points in [250, 500, 100]:
    #    for threshold in [0.05, 0.07, 0.09]:
    #        for window_size in ["1d", "2d", "5d"]:  # fake timestamps, hence days
    #            conf = DataDriftTriggerConfig(
    #                evaluation_interval_data_points=evaluation_interval_data_points,
    #                windowing_strategy=TimeWindowingStrategy(
    #                    allow_overlap=True, limit_ref=window_size, limit_cur=window_size
    #                ),
    #                metrics={
    #                    "mmd_alibi": AlibiDetectMmdDriftMetric(
    #                        device="cpu",
    #                        num_permutations=None,
    #                        decision_criterion=ThresholdDecisionCriterion(threshold=threshold),
    #                     )
    #                },
    #            )
    #           name = f"mmdalibi_{evaluation_interval_data_points}_{threshold}_{window_size}"
    #           strategies.append((name, conf))

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
        for triggering_strategy_id, triggering_strategy in gen_triggering_strategies(train_gpu):
            if (
                isinstance(triggering_strategy, DataDriftTriggerConfig)
                or isinstance(triggering_strategy, PerformanceTriggerConfig)
            ) and seed != seeds[0]:
                continue  # only execute drift triggers once

            pipeline_config = gen_arxiv_triggering_config(triggering_strategy_id, train_gpu, triggering_strategy, seed)

            if run_id % num_gpus == gpu_id and (pipeline_config.pipeline.name, seed) not in existing_pipelines:
                logger.info(f"Running {triggering_strategy_id} with seed {seed} on this GPU.")
                pipeline_configs.append(pipeline_config)

            run_id += 1

    host = os.getenv("MODYN_SUPERVISOR_HOST")
    port = os.getenv("MODYN_SUPERVISOR_PORT")

    if not host:
        host = input("Enter the supervisors host address: ") or "localhost"
    if not port:
        port = int(input("Enter the supervisors port: ") or "3000")

    run_multiple_pipelines(
        client_config=ModynClientConfig(supervisor=Supervisor(ip=host, port=port)),
        pipeline_configs=pipeline_configs,
        start_replay_at=0,
        stop_replay_at=None,
        maximum_triggers=None,
        show_eval_progress=False,
    )


if __name__ == "__main__":
    run_experiment()
