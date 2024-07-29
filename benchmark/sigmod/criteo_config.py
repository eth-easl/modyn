from __future__ import annotations

import logging

from modyn.config import (
    CheckpointingConfig,
    LrSchedulerConfig,
    OptimizationCriterion,
    OptimizerConfig,
    OptimizerParamGroup, TimeTriggerConfig, EvaluationConfig, EvalDataConfig, AccuracyMetricConfig, F1ScoreMetricConfig,
    RocAucMetricConfig,
)
from modyn.config.schema.pipeline import (
    DataConfig,
    FullModelStrategy,
    ModelConfig,
    ModynPipelineConfig,
    Pipeline,
    PipelineModelStorageConfig,
    SelectionStrategy,
    TrainingConfig,
)
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig
from modyn.config.schema.pipeline.evaluation.strategy.slicing import SlicingEvalStrategyConfig
from modyn.config.schema.pipeline.sampling.config import NewDataStrategyConfig
from modyn.config.schema.pipeline.trigger import DataAmountTriggerConfig

logger = logging.getLogger(__name__)


def gen_model_config() -> ModelConfig:
    return ModelConfig(
        id="DLRM",
        config={
            "embedding_dim": 128,
            "interaction_op": "cuda_dot",
            "hash_indices": False,
            "bottom_mlp_sizes": [512, 256, 128],
            "top_mlp_sizes": [1024, 1024, 512, 256, 1],
            "embedding_type": "joint_fused",
            "num_numerical_features": 13,
            "use_cpp_mlp": True,
            "categorical_features_info": {
                "cat_0": 7912889,
                "cat_1": 33823,
                "cat_2": 17139,
                "cat_3": 7339,
                "cat_4": 20046,
                "cat_5": 4,
                "cat_6": 7105,
                "cat_7": 1382,
                "cat_8": 63,
                "cat_9": 5554114,
                "cat_10": 582469,
                "cat_11": 245828,
                "cat_12": 11,
                "cat_13": 2209,
                "cat_14": 10667,
                "cat_15": 104,
                "cat_16": 4,
                "cat_17": 968,
                "cat_18": 15,
                "cat_19": 8165896,
                "cat_20": 2675940,
                "cat_21": 7156453,
                "cat_22": 302516,
                "cat_23": 12022,
                "cat_24": 97,
                "cat_25": 35,
            },
        },
    )


def gen_criteo_config(
    config_id: str,
    num_epochs: int,
    gpu_device: str,
    selection_strategy: SelectionStrategy,
    lr_scheduler: LrSchedulerConfig | None,
    model: str,
    dataset: str,
    num_classes: int,
    seed: int,
    optimizer: str,
    lr: float,
    trigger_period: str,
) -> ModynPipelineConfig:
    del lr_scheduler
    del model
    del dataset
    del num_classes
    del optimizer
    del lr
    bytes_parser_func = (
        "import torch\n"
        "def bytes_parser_function(x: memoryview) -> dict:\n"
        "   return {\n"
        '       "numerical_input": torch.frombuffer(x, dtype=torch.float32, count=13),\n'
        '       "categorical_input": torch.frombuffer(x, dtype=torch.int32, offset=52).long()\n'
        "   }"
    )

    # we need to convert our integer-type labels to floats,
    # since the BCEWithLogitsLoss function does not work with integers.
    label_transformer_function = (
        "import torch\n"
        "def label_transformer_function(x: torch.Tensor) -> torch.Tensor:\n"
        "  return x.to(torch.float32)"
    )
    dataloader_workers = 16
    num_prefetched_partitions = 1
    parallel_prefetch_requests = 1

    return ModynPipelineConfig(
        pipeline=Pipeline(
            name=f"criteo_{config_id}",
            description="Criteo throughput test",
            version="0.0.1",
        ),
        model=gen_model_config(),
        model_storage=PipelineModelStorageConfig(full_model_strategy=FullModelStrategy(name="PyTorchFullModel")),
        training=TrainingConfig(
            gpus=1,
            device=gpu_device,
            dataloader_workers=dataloader_workers,
            num_prefetched_partitions=num_prefetched_partitions,
            parallel_prefetch_requests=parallel_prefetch_requests,
            epochs_per_trigger=num_epochs,
            use_previous_model=True,
            initial_model="random",
            batch_size=65536,
            optimizers=[
                OptimizerConfig(
                    name="mlp",
                    algorithm="FusedSGD",
                    source="APEX",
                    param_groups=[
                        OptimizerParamGroup(module="model.top_model", config={"lr": 24}),
                        OptimizerParamGroup(module="model.bottom_model.mlp", config={"lr": 24}),
                    ],
                ),
                OptimizerConfig(
                    name="opt_1",
                    algorithm="SGD",
                    source="PyTorch",
                    param_groups=[OptimizerParamGroup(module="model.bottom_model.embeddings", config={"lr": 24})],
                ),
            ],
            optimization_criterion=OptimizationCriterion(name="BCEWithLogitsLoss"),
            checkpointing=CheckpointingConfig(activated=False),
            lr_scheduler=LrSchedulerConfig(
                name="DLRMScheduler",
                source="Custom",
                optimizers=["mlp", "opt_1"],
                step_every="batch",
                config={
                    "base_lrs": [[24, 24], [24]],
                    "warmup_steps": 8000,
                    "warmup_factor": 0,
                    "decay_steps": 24000,
                    "decay_start_step": 48000,
                    "decay_power": 2,
                    "end_lr_factor": 0,
                },
            ),
            shuffle=True,
            amp=True,
            seed=seed,
            grad_scaler_config={"growth_interval": 1000000000},
        ),
        selection_strategy=selection_strategy,
        data=DataConfig(
            dataset_id="criteo_train",
            label_transformer_function=label_transformer_function,
            bytes_parser_function=bytes_parser_func,
        ),
        trigger=TimeTriggerConfig(every=trigger_period),
        evaluation=EvaluationConfig(
            handlers=[
                EvalHandlerConfig(
                    name="exactmatrix",
                    execution_time="after_pipeline",
                    models="matrix",
                    datasets=["criteo_test"],
                    strategy=SlicingEvalStrategyConfig(eval_every="1d", eval_start_from=0, eval_end_at=86400 * 10),
                )
            ],
            after_pipeline_evaluation_workers=1,
            device=gpu_device,
            datasets=[
                EvalDataConfig(
                    dataset_id="criteo_test",
                    bytes_parser_function=bytes_parser_func,
                    batch_size=65536,
                    dataloader_workers=dataloader_workers,
                    metrics=[
                        AccuracyMetricConfig(
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "    return torch.ge(torch.sigmoid(model_output).float(), 0.5)"
                            ),
                            topn=1,
                        ),
                        RocAucMetricConfig(
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "    return torch.sigmoid(model_output).float()"
                            ),
                        ),
                        F1ScoreMetricConfig(
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.round(model_output).long()"
                            ),
                            num_classes=2,
                            average="binary",
                            pos_label=1,
                        ),
                        F1ScoreMetricConfig(
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.round(model_output).long()"
                            ),
                            num_classes=2,
                            average="binary",
                            pos_label=0,
                        ),
                        F1ScoreMetricConfig(
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.round(model_output).long()"
                            ),
                            num_classes=2,
                            average="macro",
                        ),
                        F1ScoreMetricConfig(
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.round(model_output).long()"
                            ),
                            num_classes=2,
                            average="weighted",
                        ),
                    ],
                )
            ]
        )
    )
