from __future__ import annotations

import logging

from modyn.config import CheckpointingConfig, OptimizationCriterion, OptimizerConfig, OptimizerParamGroup
from modyn.config.schema.pipeline import (
    DataConfig,
    FullModelStrategy,
    ModelConfig,
    ModynPipelineConfig,
    Pipeline,
    PipelineModelStorageConfig,
    TrainingConfig,
)
from modyn.config.schema.pipeline.sampling.config import NewDataStrategyConfig
from modyn.config.schema.pipeline.trigger import DataAmountTriggerConfig

logger = logging.getLogger(__name__)


def gen_cglm_tput_config(
    gpu_device: str,
    seed: int,
    dataloader_workers: int,
    num_prefetched_partitions: int,
    parallel_prefetch_requests: int,
    shuffle: bool,
    partition_size: int,
) -> ModynPipelineConfig:
    model_config = ModelConfig(id="ResNet50", config={"use_pretrained": False, "num_classes": 6404})

    bytes_parser_func = (
        "from PIL import Image\n"
        "import io\n"
        "def bytes_parser_function(data: memoryview) -> Image:\n"
        "   return Image.open(io.BytesIO(data)).convert('RGB')"
    )

    transformations = [
        "transforms.RandomResizedCrop(224)",
        "transforms.RandomHorizontalFlip()",
        "transforms.ToTensor()",
        "transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
    ]

    return ModynPipelineConfig(
        pipeline=Pipeline(
            name=f"cglm_{dataloader_workers}_{num_prefetched_partitions}_{parallel_prefetch_requests}_{partition_size}_{shuffle}",
            description="CGLM throughput config",
            version="0.0.1",
        ),
        model=model_config,
        model_storage=PipelineModelStorageConfig(full_model_strategy=FullModelStrategy(name="PyTorchFullModel")),
        training=TrainingConfig(
            gpus=1,
            device=gpu_device,
            dataloader_workers=dataloader_workers,
            num_prefetched_partitions=num_prefetched_partitions,
            parallel_prefetch_requests=parallel_prefetch_requests,
            use_previous_model=True,
            initial_model="random",
            batch_size=256,
            optimizers=[
                OptimizerConfig(
                    name="default",
                    algorithm="SGD",
                    source="PyTorch",
                    param_groups=[
                        OptimizerParamGroup(
                            module="model", config={"lr": 0.025, "momentum": 0.9, "weight_decay": 0.0001}
                        )
                    ],
                )
            ],
            optimization_criterion=OptimizationCriterion(name="CrossEntropyLoss"),
            checkpointing=CheckpointingConfig(activated=False),
            epochs_per_trigger=1,
            shuffle=shuffle,
            amp=False,
            seed=seed,
        ),
        selection_strategy=NewDataStrategyConfig(
            maximum_keys_in_memory=partition_size, storage_backend="local", tail_triggers=0, limit=-1
        ),
        data=DataConfig(
            dataset_id="cglm_landmark_min25", transformations=transformations, bytes_parser_function=bytes_parser_func
        ),
        trigger=DataAmountTriggerConfig(num_samples=500000),
    )
