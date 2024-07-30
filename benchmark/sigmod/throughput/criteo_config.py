from __future__ import annotations

import logging

from modyn.config import (
    CheckpointingConfig,
    LrSchedulerConfig,
    OptimizationCriterion,
    OptimizerConfig,
    OptimizerParamGroup,
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
    gpu_device: str,
    seed: int,
    dataloader_workers: int,
    num_prefetched_partitions: int,
    parallel_prefetch_requests: int,
    shuffle: bool,
    partition_size: int,
) -> ModynPipelineConfig:
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

    return ModynPipelineConfig(
        pipeline=Pipeline(
            name=f"criteo_{dataloader_workers}_{num_prefetched_partitions}_{parallel_prefetch_requests}_{partition_size}_{shuffle}",
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
            epochs_per_trigger=1,
            shuffle=shuffle,
            amp=True,
            seed=seed,
            grad_scaler_config={"growth_interval": 1000000000},
        ),
        selection_strategy=NewDataStrategyConfig(
            maximum_keys_in_memory=partition_size, storage_backend="local", tail_triggers=0, limit=-1
        ),
        data=DataConfig(
            dataset_id="criteo_tiny",
            label_transformer_function=label_transformer_function,
            bytes_parser_function=bytes_parser_func,
        ),
        trigger=DataAmountTriggerConfig(num_samples=30000000),
    )
