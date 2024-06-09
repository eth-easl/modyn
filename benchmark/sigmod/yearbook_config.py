from __future__ import annotations

from modyn.config.schema.pipeline import (
    DataConfig,
    EvalDataConfig,
    EvaluationConfig,
    FullModelStrategy,
    MatrixEvalStrategyConfig,
    MatrixEvalStrategyModel,
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


def gen_yearbook_config(
    config_id: str,
    num_epochs: int,
    gpu_device: str,
    selection_strategy: SelectionStrategy,
    lr_scheduler: LrSchedulerConfig | None,
) -> ModynPipelineConfig:
    return ModynPipelineConfig(
        pipeline=Pipeline(name=f"yearbook_{config_id}", description="Yearbook data selection config", version="0.0.1"),
        model=ModelConfig(id="YearbookNet", config={"num_input_channels": 3, "num_classes": 2}),
        model_storage=PipelineModelStorageConfig(full_model_strategy=FullModelStrategy(name="PyTorchFullModel")),
        training=TrainingConfig(
            gpus=1,
            device=gpu_device,
            dataloader_workers=1,
            use_previous_model=True,
            initial_model="random",
            batch_size=64,  # TODO(MaxiBoether): Do we want to increase this? Might affect BtS.
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
        ),
        selection_strategy=selection_strategy,
        data=DataConfig(
            dataset_id="yearbook",
            transformations=[],
            bytes_parser_function=(
                "import warnings\n"
                "import torch\n"
                "def bytes_parser_function(data: memoryview) -> torch.Tensor:\n"
                "    with warnings.catch_warnings():\n"
                "       warnings.simplefilter('ignore', category=UserWarning)\n"
                "       return torch.frombuffer(data, dtype=torch.float32).reshape(3, 32, 32)"
            ),
        ),
        trigger=TimeTriggerConfig(every="1d"),
        evaluation=EvaluationConfig(
            eval_strategy=MatrixEvalStrategyModel(
                name="MatrixEvalStrategy",
                config=MatrixEvalStrategyConfig(eval_every="1d", eval_start_from=0, eval_end_at=7258000),
            ),
            device=gpu_device,
            result_writers=["json"],
            datasets=[
                EvalDataConfig(
                    dataset_id="yearbook-test",
                    bytes_parser_function=(
                        "import warnings\n"
                        "import torch\n"
                        "def bytes_parser_function(data: memoryview) -> torch.Tensor:\n"
                        "    with warnings.catch_warnings():\n"
                        "       warnings.simplefilter('ignore', category=UserWarning)\n"
                        "       return torch.frombuffer(data, dtype=torch.float32).reshape(3, 32, 32)"
                    ),
                    batch_size=64,
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
                ),
                EvalDataConfig(
                    dataset_id="yearbook",
                    bytes_parser_function=(
                        "import warnings\n"
                        "import torch\n"
                        "def bytes_parser_function(data: memoryview) -> torch.Tensor:\n"
                        "    with warnings.catch_warnings():\n"
                        "       warnings.simplefilter('ignore', category=UserWarning)\n"
                        "       return torch.frombuffer(data, dtype=torch.float32).reshape(3, 32, 32)"
                    ),
                    batch_size=64,
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
                ),
            ],
        ),
    )
