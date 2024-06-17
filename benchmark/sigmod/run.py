from __future__ import annotations

import os
import logging
import sys

from experiments.utils.experiment_runner import run_multiple_pipelines
from benchmark.sigmod.yearbook_config import gen_yearbook_config, gen_yearbook_training_conf
from benchmark.sigmod.arxiv_config import gen_arxiv_config, gen_arxiv_training_conf
from modyn.config.schema.pipeline.sampling.downsampling_config import ILTrainingConfig
from modyn.config.schema.pipeline.training.config import TrainingConfig
from modyn.utils.utils import current_time_millis

from benchmark.sigmod.cglm_config import gen_cglm_config, gen_cglm_training_conf
from modyn.config.schema.pipeline import ModynPipelineConfig
from modyn.config import LrSchedulerConfig

from modyn.config.schema.pipeline import (
    SelectionStrategy,
    NewDataStrategyConfig,
    CoresetStrategyConfig,
    PresamplingConfig,
)
from modyn.config import (
    RS2DownsamplingConfig,
    LossDownsamplingConfig,
    GradNormDownsamplingConfig,
    RHOLossDownsamplingConfig,
    UncertaintyDownsamplingConfig,
)
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
    warmup_triggers: int, num_classes: int, training_config: TrainingConfig
) -> list[tuple[str, SelectionStrategy]]:
    strategies = []

    # Full data training
    strategies.append(
        (
            "full",
            NewDataStrategyConfig(maximum_keys_in_memory=100000, storage_backend="database", tail_triggers=0, limit=-1),
        )
    )

    # Uniform random sampling
    strategies.append(
        (
            "uniform",
            CoresetStrategyConfig(
                maximum_keys_in_memory=100000,
                storage_backend="database",
                tail_triggers=0,
                limit=-1,
                warmup_triggers=warmup_triggers,
                presampling_config=PresamplingConfig(strategy="Random", ratio=50),
            ),
        )
    )

    # Class balanced sampling
    strategies.append(
        (
            "classb",
            CoresetStrategyConfig(
                maximum_keys_in_memory=100000,
                storage_backend="database",
                tail_triggers=0,
                limit=-1,
                warmup_triggers=warmup_triggers,
                presampling_config=PresamplingConfig(strategy="LabelBalanced", ratio=50),
            ),
        )
    )

    # RS2 with replacement
    strategies.append(
        (
            "rs2w",
            CoresetStrategyConfig(
                maximum_keys_in_memory=100000,
                storage_backend="database",
                tail_triggers=0,
                limit=-1,
                warmup_triggers=warmup_triggers,
                downsampling_config=RS2DownsamplingConfig(ratio=50, with_replacement=True),
            ),
        )
    )
    # RS2 without replacement
    strategies.append(
        (
            "rs2wo",
            CoresetStrategyConfig(
                maximum_keys_in_memory=100000,
                storage_backend="database",
                tail_triggers=0,
                limit=-1,
                warmup_triggers=warmup_triggers,
                downsampling_config=RS2DownsamplingConfig(ratio=50, with_replacement=False),
            ),
        )
    )
    # Loss StB every epoch
    strategies.append(
        (
            "loss_stb",
            CoresetStrategyConfig(
                maximum_keys_in_memory=100000,
                storage_backend="database",
                tail_triggers=0,
                limit=-1,
                warmup_triggers=warmup_triggers,
                downsampling_config=LossDownsamplingConfig(ratio=50, sample_then_batch=True, period=1),
            ),
        )
    )

    # Loss BtS
    strategies.append(
        (
            "loss_bts",
            CoresetStrategyConfig(
                maximum_keys_in_memory=100000,
                storage_backend="database",
                tail_triggers=0,
                limit=-1,
                warmup_triggers=warmup_triggers,
                downsampling_config=LossDownsamplingConfig(ratio=50, sample_then_batch=False),
            ),
        )
    )

    # Gradnorm StB every epoch
    strategies.append(
        (
            "grad_stb",
            CoresetStrategyConfig(
                maximum_keys_in_memory=100000,
                storage_backend="database",
                tail_triggers=0,
                limit=-1,
                warmup_triggers=warmup_triggers,
                downsampling_config=GradNormDownsamplingConfig(ratio=50, sample_then_batch=True, period=1),
            ),
        )
    )

    # Gradnorm BtS
    strategies.append(
        (
            "grad_bts",
            CoresetStrategyConfig(
                maximum_keys_in_memory=100000,
                storage_backend="database",
                tail_triggers=0,
                limit=-1,
                warmup_triggers=warmup_triggers,
                downsampling_config=GradNormDownsamplingConfig(ratio=50, sample_then_batch=False),
            ),
        )
    )

    # RHOLossDownsamplingConfig
    il_config_options = {
        "il_model_id": "ResNet18",
        "il_model_config": {"use_pretrained": True, "num_classes": num_classes},
        "use_previous_model": False,
    }

    training_config_dict = training_config.model_dump()
    training_config_dict.update(il_config_options)
    strategies.append(
        (
            "rho_loss_bts_10il",
            CoresetStrategyConfig(
                maximum_keys_in_memory=100000,
                storage_backend="database",
                tail_triggers=0,
                limit=-1,
                warmup_triggers=warmup_triggers,
                downsampling_config=RHOLossDownsamplingConfig(
                    ratio=50,
                    sample_then_batch=False,
                    period=1,
                    holdout_set_ratio=10,
                    il_training_config=ILTrainingConfig(**training_config_dict),
                ),
            ),
        )
    )

    # Margin StB every epoch
    strategies.append(
        (
            "margin_stb",
            CoresetStrategyConfig(
                maximum_keys_in_memory=100000,
                storage_backend="database",
                tail_triggers=0,
                limit=-1,
                warmup_triggers=warmup_triggers,
                downsampling_config=UncertaintyDownsamplingConfig(
                    ratio=50, sample_then_batch=True, period=1, score_metric="Margin"
                ),
            ),
        )
    )

    # Margin BtS
    strategies.append(
        (
            "margin_bts",
            CoresetStrategyConfig(
                maximum_keys_in_memory=100000,
                storage_backend="database",
                tail_triggers=0,
                limit=-1,
                warmup_triggers=warmup_triggers,
                downsampling_config=UncertaintyDownsamplingConfig(
                    ratio=50, sample_then_batch=False, period=1, score_metric="Margin"
                ),
            ),
        )
    )

    # LeastConf StB every epoch
    strategies.append(
        (
            "lc_stb",
            CoresetStrategyConfig(
                maximum_keys_in_memory=100000,
                storage_backend="database",
                tail_triggers=0,
                limit=-1,
                warmup_triggers=warmup_triggers,
                downsampling_config=UncertaintyDownsamplingConfig(
                    ratio=50, sample_then_batch=True, period=1, score_metric="LeastConfidence"
                ),
            ),
        )
    )

    # LeastConf BtS
    strategies.append(
        (
            "lc_bts",
            CoresetStrategyConfig(
                maximum_keys_in_memory=100000,
                storage_backend="database",
                tail_triggers=0,
                limit=-1,
                warmup_triggers=warmup_triggers,
                downsampling_config=UncertaintyDownsamplingConfig(
                    ratio=50, sample_then_batch=False, period=1, score_metric="LeastConfidence"
                ),
            ),
        )
    )

    # Entropy StB every epoch
    strategies.append(
        (
            "entropy_stb",
            CoresetStrategyConfig(
                maximum_keys_in_memory=100000,
                storage_backend="database",
                tail_triggers=0,
                limit=-1,
                warmup_triggers=warmup_triggers,
                downsampling_config=UncertaintyDownsamplingConfig(
                    ratio=50, sample_then_batch=True, period=1, score_metric="Entropy"
                ),
            ),
        )
    )

    # Entropy BtS
    strategies.append(
        (
            "entropy_bts",
            CoresetStrategyConfig(
                maximum_keys_in_memory=100000,
                storage_backend="database",
                tail_triggers=0,
                limit=-1,
                warmup_triggers=warmup_triggers,
                downsampling_config=UncertaintyDownsamplingConfig(
                    ratio=50, sample_then_batch=False, period=1, score_metric="Entropy"
                ),
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

    # TODO(MaxiBoether): Implement / test learning rate warmup

    return configs


def run_experiment() -> None:
    logger.info("Grüeziwohl!")
    pipeline_configs: list[ModynPipelineConfig] = []

    pipeline_gen_func = gen_yearbook_config  # gen_arxiv_config
    # pipeline_gen_func = gen_arxiv_config
    # pipeline_gen_func = gen_cglm_config

    dataset = "cglm_landmark_min25"  # necessary for CGLM, ignored for others
    train_gpu = "cuda:0"
    num_epochs = 5  # default value, for CGLM/arxiv/yearbook see below
    warmup_triggers = 1  # default value, for CGLM/arxiv/yearbook see below
    disable_scheduling = True  # For our baselines, scheduling was mostly meaningless.
    seed = 42  # set to None to disable, should be 0-100
    num_gpus = 1  # to parallelize across gpus
    gpu_id = 0

    ## only touch if sure you wanna touch
    model = "yearbooknet"  # necessary for yearbook, ignored for others
    optimizer = None  # ignored for non arxiv
    lr = None  # ignored for non arxiv
    num_classes = 6404  # necessary for CGLM, ignored for others
    train_conf_func = None
    if pipeline_gen_func == gen_yearbook_config:
        min_lr = 1e-4
        warmup_triggers = 2
        num_epochs = 5
        optimizer = "SGD"
        config_str_fn = (
            lambda model,
            selection_strategy_id,
            lr_sched_id,
            num_epochs,
            warmup_triggers,
            dataset: f"{model}_{selection_strategy_id}_{lr_sched_id}_epoch{num_epochs}_warm{warmup_triggers}"
        )
        train_conf_func = gen_yearbook_training_conf
    elif pipeline_gen_func == gen_arxiv_config:
        min_lr = 0.00001
        warmup_triggers = 1
        num_epochs = 10  # OR 5??? OR 15??
        optimizer = "SGD"  # alternative: AdamW
        lr = 0.00002  # alternative: 0.00005
        config_str_fn = (
            lambda model,
            selection_strategy_id,
            lr_sched_id,
            num_epochs,
            warmup_triggers,
            dataset: f"{selection_strategy_id}_{lr_sched_id}_epoch{num_epochs}_warm{warmup_triggers}"
        )
        train_conf_func = gen_arxiv_training_conf

    elif pipeline_gen_func == gen_cglm_config:
        min_lr = 0.0025
        warmup_triggers = 5
        num_epochs = 5
        optimizer = "CGLM"
        config_str_fn = (
            lambda model,
            selection_strategy_id,
            lr_sched_id,
            num_epochs,
            warmup_triggers,
            dataset: f"{selection_strategy_id}_{lr_sched_id}_epoch{num_epochs}_warm{warmup_triggers}_ds{dataset}"
        )
        ds_class_map = {"cglm_landmark_min25": 6404, "cglm_hierarchical_min25": 79}
        num_classes = ds_class_map[dataset]
        train_conf_func = gen_cglm_training_conf

    run_id = 0
    for lr_sched_id, lr_scheduler_config in gen_lr_scheduler_configs(min_lr, disable_scheduling):
        train_conf = train_conf_func(optimizer, lr, train_gpu, lr_scheduler_config, num_epochs, seed)
        for selection_strategy_id, selection_strategy in gen_selection_strategies(
            warmup_triggers, num_classes, train_conf
        ):
            config_id = config_str_fn(model, selection_strategy_id, lr_sched_id, num_epochs, warmup_triggers, dataset)
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
        maximum_triggers=None,
        show_eval_progress=False,
    )


if __name__ == "__main__":
    run_experiment()
