from __future__ import annotations

from modyn.config import (
    CheckpointingConfig,
    LrSchedulerConfig,
    OptimizationCriterion,
    OptimizerConfig,
    OptimizerParamGroup,
)
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
    SelectionStrategy,
    TimeTriggerConfig,
    TrainingConfig,
)
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig
from modyn.config.schema.pipeline.evaluation.strategy.periodic import PeriodicEvalStrategyConfig
from modyn.config.schema.pipeline.evaluation.strategy.slicing import SlicingEvalStrategyConfig


def gen_yearbook_training_conf(
    optimizer: str, lr: float, gpu_device: str, lr_scheduler: LrSchedulerConfig | None, num_epochs: int, seed: int
):
    assert optimizer == "SGD"
    del lr
    return TrainingConfig(
        gpus=1,
        device=gpu_device,
        dataloader_workers=1,
        use_previous_model=True,
        initial_model="random",
        batch_size=64,
        optimizers=[
            OptimizerConfig(
                name="default",
                algorithm="SGD",
                source="PyTorch",
                param_groups=[OptimizerParamGroup(module="model", config={"lr": 0.001, "momentum": 0.9})],
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


def _yearbook_model(model: str) -> tuple[ModelConfig, str, list]:
    if model.lower() == "yearbooknet":
        model_config = ModelConfig(id="YearbookNet", config={"num_input_channels": 3, "num_classes": 2})
        parser_func = (
            "import warnings\n"
            "import torch\n"
            "def bytes_parser_function(data: memoryview) -> torch.Tensor:\n"
            "    with warnings.catch_warnings():\n"
            "       warnings.simplefilter('ignore', category=UserWarning)\n"
            "       return torch.frombuffer(data, dtype=torch.float32).reshape(3, 32, 32)"
        )

        return model_config, parser_func, []

    if model.lower() == "resnet18":
        model_config = ModelConfig(id="ResNet18", config={"use_pretrained": True, "num_classes": 2})
        parser_func = (
            "from PIL import Image\n"
            "import io\n"
            "def bytes_parser_function(data: memoryview) -> Image:\n"
            "   return Image.frombuffer('RGB', (32, 32), data).convert('RGB')"
        )
        transformations = [
            "transforms.ToTensor()",
            "transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
        ]

        return model_config, parser_func, transformations

    raise ValueError(f"Unknown model: {model}")


def gen_yearbook_config(
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
    del dataset
    del num_classes
    del optimizer
    del lr
    model_config, bytes_parser_func, transformations = _yearbook_model(model)
    return ModynPipelineConfig(
        pipeline=Pipeline(name=f"yearbook_{config_id}", description="Yearbook data selection config", version="0.0.1"),
        model=model_config,
        model_storage=PipelineModelStorageConfig(full_model_strategy=FullModelStrategy(name="PyTorchFullModel")),
        training=gen_yearbook_training_conf("SGD", 0.42, gpu_device, lr_scheduler, num_epochs, seed),
        selection_strategy=selection_strategy,
        data=DataConfig(
            dataset_id="yearbook", transformations=transformations, bytes_parser_function=bytes_parser_func
        ),
        trigger=TimeTriggerConfig(every="1d"),
        evaluation=EvaluationConfig(
            handlers=[
                EvalHandlerConfig(
                    name="exactmatrix",
                    execution_time="after_training",
                    models="matrix",
                    datasets=["yearbook-test"],
                    strategy=SlicingEvalStrategyConfig(eval_every="1d", eval_start_from=0, eval_end_at=7258000),
                ),
                EvalHandlerConfig(
                    name="slidingmatrix",
                    execution_time="after_training",
                    models="matrix",
                    datasets=["yearbook-test"],
                    strategy=PeriodicEvalStrategyConfig(
                        every="1d", interval="[-25h; +25h]", start_timestamp=0, end_timestamp=1400000
                    ),
                ),
            ],
            after_pipeline_evaluation_workers=12,
            after_training_evaluation_workers=12,
            device=gpu_device,
            result_writers=["json"],
            datasets=[
                EvalDataConfig(
                    dataset_id=dataset,
                    bytes_parser_function=bytes_parser_func,
                    transformations=transformations,
                    batch_size=512,
                    dataloader_workers=1,
                    metrics=[
                        Metric(
                            name="Accuracy",
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "    return torch.argmax(model_output, dim=-1)"
                            ),
                            config={"num_classes": 2},
                        ),
                        Metric(
                            name="F1Score",
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.argmax(model_output, dim=-1)"
                            ),
                            config={"num_classes": 2, "average": "weighted"},
                        ),
                        Metric(
                            name="F1Score",
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.argmax(model_output, dim=-1)"
                            ),
                            config={"num_classes": 2, "average": "macro"},
                        ),
                        Metric(
                            name="F1Score",
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.argmax(model_output, dim=-1)"
                            ),
                            config={"num_classes": 2, "average": "micro"},
                        ),
                    ],
                )
                for dataset in ["yearbook", "yearbook-test"]
            ],
        ),
    )
