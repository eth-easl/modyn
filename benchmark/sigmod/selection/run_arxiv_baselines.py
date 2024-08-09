from __future__ import annotations

import logging
import os
import sys

from benchmark.sigmod.arxiv_config import gen_arxiv_config
from experiments.utils.experiment_runner import run_multiple_pipelines
from modyn.config import LrSchedulerConfig
from modyn.config.schema.pipeline import (
    ModynPipelineConfig,
    NewDataStrategyConfig,
    SelectionStrategy,
)
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


def gen_selection_strategies(
    warmup_triggers: int,
    num_classes: int,
) -> list[tuple[str, SelectionStrategy]]:
    strategies = []

    # Full data training
    strategies.append(
        (
            "full",
            NewDataStrategyConfig(
                maximum_keys_in_memory=100000,
                storage_backend="database",
                tail_triggers=0,
                limit=-1,
            ),
        )
    )

    return strategies


def gen_lr_scheduler_configs(min_lr: float, disable: bool) -> list[tuple[str, None | LrSchedulerConfig]]:
    configs = []

    # No LR scheduling
    configs.append(("nosched", None))

    if disable:
        return configs

    # Cosine scheduling
    configs.append(
        (
            "cosine",
            LrSchedulerConfig(
                name="CosineAnnealingLR",
                source="PyTorch",
                step_every="batch",
                optimizers=["default"],
                config={"T_max": "MODYN_NUM_BATCHES", "eta_min": min_lr},
            ),
        )
    )

    return configs


def run_experiment() -> None:
    logger.info("Grüeziwohl!")
    pipeline_configs: list[ModynPipelineConfig] = []

    pipeline_gen_func = gen_arxiv_config

    dataset = "arxiv"  # necessary for CGLM, ignored for others
    train_gpu = "cuda:0"
    warmup_triggers = 1  # default value, for CGLM/arxiv/yearbook see below
    disable_scheduling = False  # For our baselines, scheduling was mostly meaningless.
    seed = 42  # set to None to disable, should be 0-100
    num_gpus = 1  # to parallelize across gpus
    gpu_id = 0

    ## only touch if sure you wanna touch
    model = "yearbooknet"  # necessary for yearbook, ignored for others
    optimizer = None  # ignored for non arxiv
    lr = None  # ignored for non arxiv
    num_classes = 6404  # necessary for CGLM, ignored for others

    min_lr = 0.00001
    warmup_triggers = 1
    num_epochss = [5, 10]
    optimizers = ["AdamW", "SGD"]
    lrs = [0.00002, 0.00005]

    def config_str_fn(
        model: str,
        selection_strategy_id: str,
        lr_sched_id: str,
        num_epochs: int,
        warmup_triggers: int,
        dataset: str,
        optimizer: str,
        lr: float,
    ) -> str:
        return (
            f"{model}_{selection_strategy_id}_{lr_sched_id}_{num_epochs}_{warmup_triggers}_{dataset}_{optimizer}_{lr}"
        )

    run_id = 0
    for lr in lrs:
        for optimizer in optimizers:
            for num_epochs in num_epochss:
                for lr_sched_id, lr_scheduler_config in gen_lr_scheduler_configs(min_lr, disable_scheduling):
                    for (
                        selection_strategy_id,
                        selection_strategy,
                    ) in gen_selection_strategies(warmup_triggers, num_classes):
                        config_id = config_str_fn(
                            model,
                            selection_strategy_id,
                            lr_sched_id,
                            num_epochs,
                            warmup_triggers,
                            dataset,
                            optimizer,
                            lr,
                        )
                        if run_id % num_gpus == gpu_id:
                            pipeline_configs.append(
                                pipeline_gen_func(
                                    config_id,
                                    num_epochs,
                                    train_gpu,
                                    selection_strategy,
                                    lr_scheduler_config,
                                    model,
                                    dataset,
                                    num_classes,
                                    seed,
                                    optimizer,
                                    lr,
                                )
                            )

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
        maximum_triggers=5,
        show_eval_progress=False,
    )


if __name__ == "__main__":
    run_experiment()
