from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Annotated

import typer

from benchmark.sigmod.arxiv_config import gen_arxiv_config, gen_arxiv_training_conf
from benchmark.sigmod.cglm_config import gen_cglm_config, gen_cglm_training_conf
from benchmark.sigmod.yearbook_config import gen_yearbook_config, gen_yearbook_training_conf
from experiments.utils.experiment_runner import run_multiple_pipelines, run_multiple_pipelines_parallel
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
        # logging.FileHandler(f"client_{current_time_millis()}.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def gen_selection_strategies(
    ratio: int,
    ratio_max: int,
    warmup_triggers: int,
    num_classes: int,
    training_config: TrainingConfig,
    period: int,
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
                    maximum_keys_in_memory=100000, storage_backend="database", tail_triggers=0, limit=-1
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
                downsampling_config=RS2DownsamplingConfig(ratio=ratio, ratio_max=ratio_max,  with_replacement=True),
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
                downsampling_config=RS2DownsamplingConfig(ratio=ratio, ratio_max=ratio_max,  with_replacement=False),
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
                downsampling_config=LossDownsamplingConfig(ratio=ratio, ratio_max=ratio_max,  sample_then_batch=False),
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
                downsampling_config=GradNormDownsamplingConfig(ratio=ratio, ratio_max=ratio_max,  sample_then_batch=False),
            ),
        )
    )

    # # RHOLossDownsamplingConfig
    # for use_previous_model in [True]:
    #     for use_pretrained in [False, True]:
    #         il_config_options = {
    #             "il_model_id": "ResNet18",
    #             "il_model_config": {"use_pretrained": use_pretrained, "num_classes": num_classes},
    #             "use_previous_model": use_previous_model,
    #             "drop_last_batch": False,
    #         }
    #         if not use_pretrained:
    #             # delete the key
    #             del il_config_options["il_model_config"]["use_pretrained"]
    #         training_config_dict = training_config.model_dump()
    #         training_config_dict.update(il_config_options)
    #         epochs_per_trigger = training_config_dict["epochs_per_trigger"]
    #         use_prev_suffix = "_use_prev" if use_previous_model else ""
    #         use_pretrained_suffix = "_no_pretrained" if not use_pretrained else ""
    #         rho_name = f"rho_loss_bts_twin_{epochs_per_trigger}ep{use_prev_suffix}{use_pretrained_suffix}"
    #         strategies.append(
    #             (
    #                 rho_name,
    #                 CoresetStrategyConfig(
    #                     maximum_keys_in_memory=100000,
    #                     storage_backend="database",
    #                     tail_triggers=0,
    #                     limit=-1,
    #                     warmup_triggers=warmup_triggers,
    #                     downsampling_config=RHOLossDownsamplingConfig(
    #                         ratio=ratio, ratio_max=ratio_max,
    #                         sample_then_batch=False,
    #                         period=period,
    #                         holdout_set_ratio=50,
    #                         holdout_set_strategy="Twin",
    #                         il_training_config=ILTrainingConfig(**training_config_dict),
    #                     ),
    #                 ),
    #             )
    #         )


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
                    ratio=ratio, ratio_max=ratio_max,  sample_then_batch=False, score_metric="Margin"
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
                    ratio=ratio, ratio_max=ratio_max,  sample_then_batch=False, score_metric="LeastConfidence"
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
                    ratio=ratio, ratio_max=ratio_max,  sample_then_batch=False, score_metric="Entropy"
                ),
            ),
        )
    )

    if not small_run:
        # Gradnorm StB
        strategies.append(
            (
                f"grad_stb_period{period}",
                CoresetStrategyConfig(
                    maximum_keys_in_memory=100000,
                    storage_backend="database",
                    tail_triggers=0,
                    limit=-1,
                    warmup_triggers=warmup_triggers,
                    downsampling_config=GradNormDownsamplingConfig(ratio=ratio, ratio_max=ratio_max,  sample_then_batch=True, period=period),
                ),
            )
        )
        # Loss StB
        strategies.append(
            (
                f"loss_stb_period{period}",
                CoresetStrategyConfig(
                    maximum_keys_in_memory=100000,
                    storage_backend="database",
                    tail_triggers=0,
                    limit=-1,
                    warmup_triggers=warmup_triggers,
                    downsampling_config=LossDownsamplingConfig(ratio=ratio, ratio_max=ratio_max,  sample_then_batch=True, period=period),
                ),
            )
        )
        # Margin StB
        strategies.append(
            (
                f"margin_stb_period{period}",
                CoresetStrategyConfig(
                    maximum_keys_in_memory=100000,
                    storage_backend="database",
                    tail_triggers=0,
                    limit=-1,
                    warmup_triggers=warmup_triggers,
                    downsampling_config=UncertaintyDownsamplingConfig(
                        ratio=ratio, ratio_max=ratio_max,  sample_then_batch=True, period=period, score_metric="Margin"
                    ),
                ),
            )
        )
        # LeastConf StB
        strategies.append(
            (
                f"lc_stb_period{period}",
                CoresetStrategyConfig(
                    maximum_keys_in_memory=100000,
                    storage_backend="database",
                    tail_triggers=0,
                    limit=-1,
                    warmup_triggers=warmup_triggers,
                    downsampling_config=UncertaintyDownsamplingConfig(
                        ratio=ratio, ratio_max=ratio_max,  sample_then_batch=True, period=period, score_metric="LeastConfidence"
                    ),
                ),
            )
        )
        # Entropy StB
        strategies.append(
            (
                f"entropy_stb_period{period}",
                CoresetStrategyConfig(
                    maximum_keys_in_memory=100000,
                    storage_backend="database",
                    tail_triggers=0,
                    limit=-1,
                    warmup_triggers=warmup_triggers,
                    downsampling_config=UncertaintyDownsamplingConfig(
                        ratio=ratio, ratio_max=ratio_max,  sample_then_batch=True, period=period, score_metric="Entropy"
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

    return configs


def config_str_fn(
    selection_strategy_id: str,
    lr_sched_id: str,
    num_epochs: int,
    warmup_triggers: int,
    ratio: int,
    trigger_period: str,
) -> str:
    return f"{selection_strategy_id}_{lr_sched_id}_epoch{num_epochs}_warm{warmup_triggers}_trigger{trigger_period}_r{ratio}"


def run_experiment(
        gpu_id: Annotated[int, typer.Argument(help="The GPU ID to run the experiment on")],
        disable_run: Annotated[bool, typer.Option(help="If set, will only print the pipelines to run")] = False,
) -> None:
    logger.info("Grüeziwohl!")

    # Pick the line you want.
    pipeline_gen_func = gen_yearbook_config
    # pipeline_gen_func = gen_arxiv_config
    # pipeline_gen_func = gen_cglm_config

    dataset = "cglm_landmark_min25"  # necessary for CGLM, ignored for others
    train_gpu = f"cuda:{gpu_id}"
    num_gpus = 4  # to parallelize across gpus
    maximal_collocation = 4  # maximal number of pipelines to run in parallel on one GPU

    period = 0
    disable_scheduling = True  # For our baselines, scheduling was mostly meaningless.
    seeds = [42]#, 99, 12]  # set to [None] to disable, should be 0-100
    ratios = [500]# 250, 125] # 12.5%, 50%, 25% due to ratio max scaling
    ratio_max = 1000
    small_run = False
    include_full = False
    skip_existing = False
    ## only touch if sure you wanna touch
    model = "yearbooknet"  # necessary for yearbook, ignored for others
    optimizer = None  # ignored for non arxiv
    lr = None  # ignored for non arxiv
    train_conf_func = None
    maximum_triggers = None
    if pipeline_gen_func == gen_yearbook_config:
        num_classes = 2
        min_lr = 1e-4
        warmup_triggers = 2
        num_epochs = 5
        optimizer = "SGD"
        train_conf_func = gen_yearbook_training_conf
        trigger_period = "1d"
    elif pipeline_gen_func == gen_arxiv_config:
        min_lr = 0.00001
        warmup_triggers = 1
        num_epochs = 5
        num_classes = 172
        optimizer = "AdamW"
        lr = 0.00002
        train_conf_func = gen_arxiv_training_conf
        trigger_period = "1d"
    elif pipeline_gen_func == gen_cglm_config:
        min_lr = 0.0025
        warmup_triggers = 5
        num_epochs = 5
        optimizer = "SGD"
        ds_class_map = {"cglm_landmark_min25": 6404, "cglm_hierarchical_min25": 79}
        num_classes = ds_class_map[dataset]
        train_conf_func = gen_cglm_training_conf
        maximum_triggers = 17  # last triggers are meaningless and cost time
        trigger_period = "1y"
    else:
        raise RuntimeError("Unknown pipeline generator function.")

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
    pipeline_configs: list[ModynPipelineConfig] = []
    for seed in seeds:
        for ratio in ratios:
            for lr_sched_id, lr_scheduler_config in gen_lr_scheduler_configs(min_lr, disable_scheduling):
                train_conf = train_conf_func(optimizer, lr, train_gpu, lr_scheduler_config, num_epochs, seed)
                for selection_strategy_id, selection_strategy in gen_selection_strategies(
                    ratio,
                    ratio_max,
                    warmup_triggers,
                    num_classes,
                    train_conf,
                    period,
                    small_run=small_run,
                    include_full=include_full,
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
                        selection_strategy_id, lr_sched_id, num_epochs, warmup_triggers, ratio, trigger_period
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
                        trigger_period
                    )

                    if run_id % num_gpus == gpu_id and (pipeline_config.pipeline.name, seed) not in existing_pipelines:
                        logger.info(f"Running {config_id} with seed {seed} and ratio {ratio} on GPU {train_gpu}.")
                        pipeline_configs.append(pipeline_config)

                    run_id += 1

    logger.info(f"There are {run_id} pipelines to run.")
    pipeline_ids = [p.pipeline.name for p in pipeline_configs]
    logger.info(f"The current process will run: {', '.join(pipeline_ids)}")
    if disable_run:
        logger.info("Exiting without running.")
        return
    host = "localhost" #os.getenv("MODYN_SUPERVISOR_HOST")
    port = 3069 #os.getenv("MODYN_SUPERVISOR_PORT")

    # if not host:
    #     host = input("Enter the supervisors host address: ") or "localhost"
    # if not port:
    #     port = int(input("Enter the supervisors port: ") or "3000")


    run_multiple_pipelines_parallel(
        client_config=ModynClientConfig(supervisor=Supervisor(ip=host, port=port)),
        pipeline_configs=pipeline_configs,
        start_replay_at=0,
        stop_replay_at=None,
        maximum_triggers=maximum_triggers,
        show_eval_progress=False,
        maximal_collocation=maximal_collocation,
    )


if __name__ == "__main__":
    typer.run(run_experiment)
