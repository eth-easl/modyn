from __future__ import annotations

from modyn.config.schema.pipeline import (
    DataConfig,
    EvalDataConfig,
    EvaluationConfig,
    FullModelStrategy,
    Metric,
    ModelConfig,
    ModynPipelineConfig,
    Pipeline,
    PipelineModelStorageConfig,
    TimeTriggerConfig,
    TrainingConfig,
    SelectionStrategy,
)
from modyn.config import (
    CheckpointingConfig,
    OptimizationCriterion,
    OptimizerConfig,
    OptimizerParamGroup,
)
from modyn.config import LrSchedulerConfig
import logging
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig
from modyn.config.schema.pipeline.evaluation.strategy.slicing import SlicingEvalStrategyConfig

logger = logging.getLogger(__name__)

def gen_cglm_training_conf(gpu_device: str, lr_scheduler: LrSchedulerConfig | None,num_epochs: int,seed:int):
    return TrainingConfig(
                gpus=1,
                device=gpu_device,
                dataloader_workers=1,
                use_previous_model=True,
                initial_model="random",
                batch_size=128,  # TODO(MaxiBoether): Do we want to increase this? Might affect BtS.
                optimizers=[
                    OptimizerConfig(
                        name="default",
                        algorithm="SGD",
                        source="PyTorch",
                        param_groups=[OptimizerParamGroup(module="model", config={"lr": 0.005, "momentum": 0.9})],
                    )
                ],
                optimization_criterion=OptimizationCriterion(name="CrossEntropyLoss"),
                checkpointing=CheckpointingConfig(activated=False),
                lr_scheduler=lr_scheduler,
                epochs_per_trigger=num_epochs,
                shuffle=True,
                amp=False,
                seed=seed,
            )

def gen_cglm_config(
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
) -> ModynPipelineConfig:
    del model  # hardcore resnet50
    del lr
    del optimizer
    model_config = ModelConfig(id="ResNet50", config={"use_pretrained": True, "num_classes": num_classes})
    logger.debug("This is a test.")

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
        pipeline=Pipeline(name=f"cglm_{config_id}", description="CGLM data selection config", version="0.0.1"),
        model=model_config,
        model_storage=PipelineModelStorageConfig(full_model_strategy=FullModelStrategy(name="PyTorchFullModel")),
        training=gen_cglm_training_conf(gpu_device, lr_scheduler, num_epochs, seed),
        selection_strategy=selection_strategy,
        data=DataConfig(dataset_id=dataset, transformations=transformations, bytes_parser_function=bytes_parser_func),
        trigger=TimeTriggerConfig(every="1y"),
        evaluation=EvaluationConfig(
            handlers=[
                EvalHandlerConfig(
                    name="MatrixEval",
                    execution_time="after_training",
                    models="matrix",
                    datasets=[dataset, f"{dataset}-test"],
                    strategy=SlicingEvalStrategyConfig(
                        eval_every="1y", eval_start_from=1041379200, eval_end_at=1717200000
                    ),
                )
            ],
            after_pipeline_evaluation_workers=3,
            after_training_evaluation_workers=3,
            device=gpu_device,
            result_writers=["json"],
            datasets=[
                EvalDataConfig(
                    dataset_id=_dataset,
                    bytes_parser_function=bytes_parser_func,
                    transformations=transformations,
                    batch_size=256,
                    dataloader_workers=1,
                    metrics=[
                        Metric(
                            name="Accuracy",
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "    return torch.argmax(model_output, dim=-1)"
                            ),
                            config={"num_classes": num_classes},
                        ),
                        Metric(
                            name="Accuracy",
                            evaluation_transformer_function="",
                            config={"num_classes": num_classes, "topn": 2},
                        ),
                        Metric(
                            name="Accuracy",
                            evaluation_transformer_function="",
                            config={"num_classes": num_classes, "topn": 5},
                        ),
                        Metric(
                            name="F1Score",
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.argmax(model_output, dim=-1)"
                            ),
                            config={"num_classes": num_classes, "average": "weighted"},
                        ),
                        Metric(
                            name="F1Score",
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.argmax(model_output, dim=-1)"
                            ),
                            config={"num_classes": num_classes, "average": "macro"},
                        ),
                        Metric(
                            name="F1Score",
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.argmax(model_output, dim=-1)"
                            ),
                            config={"num_classes": num_classes, "average": "micro"},
                        ),
                    ],
                )
                for _dataset in [dataset, f"{dataset}-test"]
            ],
        ),
    )
