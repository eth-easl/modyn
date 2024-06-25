from __future__ import annotations

import logging
import os
from pathlib import Path
import sys

from benchmark.sigmod.throughput.cglm_config import gen_cglm_tput_config
from experiments.utils.experiment_runner import run_multiple_pipelines
from modyn.config.schema.pipeline import (
    ModynPipelineConfig,
)

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


def run_experiment() -> None:
    logger.info("Grüeziwohl!")
    pipeline_configs: list[ModynPipelineConfig] = []

    train_gpu = "cuda:0"
    seeds = [42]
    num_dataloader_workers_list = [16, 1, 4, 8]
    partition_size_list = [30000, 85000]
    num_prefetched_partitions_list = [0, 1, 2, 6]
    parallel_pref_list = [1, 2, 4, 8]
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
    for seed in seeds:
        for shuffle in [False, True]:
            for num_dataloader_workers in num_dataloader_workers_list:
                for partition_size in partition_size_list:
                    for num_prefetched_partitions in num_prefetched_partitions_list:
                        for parallel_pref in parallel_pref_list:
                            if num_prefetched_partitions == 0 and parallel_pref > 1:
                                continue

                            if num_prefetched_partitions > 0 and parallel_pref > num_prefetched_partitions:
                                continue

                            if shuffle and num_dataloader_workers not in [1, 16]:
                                continue

                            pipeline_config = gen_cglm_tput_config(
                                train_gpu,
                                seed,
                                num_dataloader_workers,
                                num_prefetched_partitions,
                                parallel_pref,
                                shuffle,
                                partition_size,
                            )

                            if (pipeline_config.pipeline.name, seed) not in existing_pipelines:
                                logger.info(f"Running {pipeline_config.pipeline.name} with seed {seed}.")
                                pipeline_configs.append(pipeline_config)

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
        maximum_triggers=1,
    )


if __name__ == "__main__":
    run_experiment()
