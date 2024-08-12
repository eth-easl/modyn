from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from benchmark.sigmod.selection.arxiv_config import (
    gen_arxiv_config,
    gen_arxiv_training_conf,
)
from benchmark.sigmod.selection.cglm_config import (
    gen_cglm_config,
    gen_cglm_training_conf,
)
from benchmark.sigmod.selection.yearbook_config import (
    gen_yearbook_config,
    gen_yearbook_training_conf,
)
from experiments.utils.experiment_runner import run_multiple_pipelines
from modyn.config import (
    GradNormDownsamplingConfig,
    LossDownsamplingConfig,
    LrSchedulerConfig,
    RHOLossDownsamplingConfig,
    RS2DownsamplingConfig,
    UncertaintyDownsamplingConfig,
)
from modyn.config.schema.pipeline import (
    CoresetStrategyConfig,
    ModynPipelineConfig,
    NewDataStrategyConfig,
    PresamplingConfig,
    SelectionStrategy,
)
from modyn.config.schema.pipeline.sampling.downsampling_config import ILTrainingConfig
from modyn.config.schema.pipeline.training.config import TrainingConfig
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


def gen_selection_strategies(
    ratio: int,
    ratio_max: int,
    warmup_triggers: int,
    num_classes: int,
    training_config: TrainingConfig,
    small_run: bool = False,
    include_full=True,
) -> list[tuple[str, SelectionStrategy]]:
    strategies = []

    if include_full:
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
                presampling_config=PresamplingConfig(strategy="Random", ratio=ratio, ratio_max=ratio_max),
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
                presampling_config=PresamplingConfig(strategy="LabelBalanced", ratio=ratio, ratio_max=ratio_max),
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
                downsampling_config=RS2DownsamplingConfig(ratio=ratio, ratio_max=ratio_max, with_replacement=True),
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
                downsampling_config=RS2DownsamplingConfig(ratio=ratio, ratio_max=ratio_max, with_replacement=False),
            ),
        )
    )
    if not small_run:
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
                    downsampling_config=LossDownsamplingConfig(
                        ratio=ratio,
                        ratio_max=ratio_max,
                        sample_then_batch=True,
                        period=1,
                    ),
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
                downsampling_config=LossDownsamplingConfig(ratio=ratio, ratio_max=ratio_max, sample_then_batch=False),
            ),
        )
    )

    if not small_run:
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
                    downsampling_config=GradNormDownsamplingConfig(
                        ratio=ratio,
                        ratio_max=ratio_max,
                        sample_then_batch=True,
                        period=1,
                    ),
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
                downsampling_config=GradNormDownsamplingConfig(
                    ratio=ratio, ratio_max=ratio_max, sample_then_batch=False
                ),
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
                    ratio=ratio,
                    ratio_max=ratio_max,
                    sample_then_batch=False,
                    period=1,
                    holdout_set_ratio=10,
                    il_training_config=ILTrainingConfig(**training_config_dict),
                ),
            ),
        )
    )

    if not small_run:
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
                        ratio=ratio,
                        ratio_max=ratio_max,
                        sample_then_batch=True,
                        period=1,
                        score_metric="Margin",
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
                    ratio=ratio,
                    ratio_max=ratio_max,
                    sample_then_batch=False,
                    period=1,
                    score_metric="Margin",
                ),
            ),
        )
    )

    if not small_run:
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
                        ratio=ratio,
                        ratio_max=ratio_max,
                        sample_then_batch=True,
                        period=1,
                        score_metric="LeastConfidence",
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
                    ratio=ratio,
                    ratio_max=ratio_max,
                    sample_then_batch=False,
                    period=1,
                    score_metric="LeastConfidence",
                ),
            ),
        )
    )

    if not small_run:
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
                        ratio=ratio,
                        ratio_max=ratio_max,
                        sample_then_batch=True,
                        period=1,
                        score_metric="Entropy",
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
                    ratio=ratio,
                    ratio_max=ratio_max,
                    sample_then_batch=False,
                    period=1,
                    score_metric="Entropy",
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

    # Pick the line you want.
    pipeline_gen_func = gen_yearbook_config
    pipeline_gen_func = gen_arxiv_config
    pipeline_gen_func = gen_cglm_config

    dataset = "cglm_landmark_min25"  # necessary for CGLM, ignored for others
    train_gpu = "cuda:0"
    num_epochs = 5  # default value, for CGLM/arxiv/yearbook see below
    warmup_triggers = 1  # default value, for CGLM/arxiv/yearbook see below
    disable_scheduling = True  # For our baselines, scheduling was mostly meaningless.
    seeds = [42, 99, 12]  # set to [None] to disable, should be 0-100
    ratios = [125, 500, 250]  # 12.5%, 50%, 25% due to ratio max scaling
    ratio_max = 1000
    num_gpus = 1  # to parallelize across gpus
    gpu_id = 0
    small_run = True
    skip_existing = True
    ## only touch if sure you wanna touch
    model = "yearbooknet"  # necessary for yearbook, ignored for others
    optimizer = None  # ignored for non arxiv
    lr = None  # ignored for non arxiv
    num_classes = 6404  # necessary for CGLM, ignored for others
    train_conf_func = None
    maximum_triggers = None
    if pipeline_gen_func == gen_yearbook_config:
        min_lr = 1e-4
        warmup_triggers = 2
        num_epochs = 5
        optimizer = "SGD"

        def config_str_fn(
            model,
            selection_strategy_id,
            lr_sched_id,
            num_epochs,
            warmup_triggers,
            ratio,
            dataset,
        ):
            return f"{model}_{selection_strategy_id}_{lr_sched_id}_epoch{num_epochs}_warm{warmup_triggers}_r{ratio}"

        train_conf_func = gen_yearbook_training_conf
    elif pipeline_gen_func == gen_arxiv_config:
        min_lr = 0.00001
        warmup_triggers = 1
        num_epochs = 5
        optimizer = "AdamW"
        lr = 0.00002

        def config_str_fn(
            model,
            selection_strategy_id,
            lr_sched_id,
            num_epochs,
            warmup_triggers,
            ratio,
            dataset,
        ):
            return f"{selection_strategy_id}_{lr_sched_id}_epoch{num_epochs}_warm{warmup_triggers}_r{ratio}"

        train_conf_func = gen_arxiv_training_conf

    elif pipeline_gen_func == gen_cglm_config:
        min_lr = 0.0025
        warmup_triggers = 5
        num_epochs = 5
        optimizer = "SGD"

        def config_str_fn(
            model: str,
            selection_strategy_id: str,
            lr_sched_id: str,
            num_epochs: int,
            warmup_triggers: int,
            ratio: int,
            dataset: str,
        ) -> str:
            return f"{model}_{selection_strategy_id}_{lr_sched_id}_epoch{num_epochs}_warm{warmup_triggers}_ds{dataset}_r{ratio}"

        ds_class_map = {"cglm_landmark_min25": 6404, "cglm_hierarchical_min25": 79}
        num_classes = ds_class_map[dataset]
        train_conf_func = gen_cglm_training_conf
        maximum_triggers = 17  # last triggers are meaningless and cost time

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
            # patch legacy names
            if name[-4] != "r":
                name = f"{name}_r500"  # we only did 50% runs before

            existing_pipelines.append((name, seed))

        logger.info(f"Found these existing pipelines: {existing_pipelines}")

    existing_pipelines = set(existing_pipelines)
    run_id = 0
    for seed in seeds:
        for ratio in ratios:
            for lr_sched_id, lr_scheduler_config in gen_lr_scheduler_configs(min_lr, disable_scheduling):
                train_conf = train_conf_func(optimizer, lr, train_gpu, lr_scheduler_config, num_epochs, seed)
                for (
                    selection_strategy_id,
                    selection_strategy,
                ) in gen_selection_strategies(
                    ratio,
                    ratio_max,
                    warmup_triggers,
                    num_classes,
                    train_conf,
                    small_run=small_run,
                    include_full=(ratio == ratios[0]),
                ):
                    if (
                        dataset == "cglm_landmark_min25"
                        and pipeline_gen_func == gen_cglm_config
                        and selection_strategy_id == "classb"
                    ):
                        continue  # classb on landmark does not work

                    if pipeline_gen_func == gen_arxiv_config and selection_strategy_id.startswith("rho_loss"):
                        continue  # we don't have a small model for RHO LOSS that deals with tokenized texts yet

                    config_id = config_str_fn(
                        model,
                        selection_strategy_id,
                        lr_sched_id,
                        num_epochs,
                        warmup_triggers,
                        ratio,
                        dataset,
                    )

                    pipeline_config = pipeline_gen_func(
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

                    if run_id % num_gpus == gpu_id and (pipeline_config.pipeline.name, seed) not in existing_pipelines:
                        logger.info(f"Running {config_id} with seed {seed} and ratio {ratio} on this GPU.")
                        pipeline_configs.append(pipeline_config)

                    run_id += 1
    # logger.info(f"Overview of configurations: {pipeline_configs}")
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
        maximum_triggers=maximum_triggers,
        show_eval_progress=False,
    )


if __name__ == "__main__":
    run_experiment()
