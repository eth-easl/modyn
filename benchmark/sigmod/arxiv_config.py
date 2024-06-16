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
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig
from modyn.config.schema.pipeline.evaluation.strategy.slicing import SlicingEvalStrategyConfig


def gen_arxiv_config(
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
    del model  # ignored for now
    del dataset
    del num_classes

    if optimizer == "SGD":
        opti_conf = OptimizerConfig(
            name="default",
            algorithm="SGD",
            source="PyTorch",
            param_groups=[
                OptimizerParamGroup(module="model", config={"lr": lr, "momentum": 0.9, "weight_decay": 0.01})
            ],
        )
    elif optimizer == "AdamW":
        opti_conf = OptimizerConfig(
            name="default",
            algorithm="AdamW",
            source="PyTorch",
            param_groups=[OptimizerParamGroup(module="model", config={"lr": lr, "weight_decay": 0.01})],
        )
    else:
        raise ValueError(optimizer)

    return ModynPipelineConfig(
        pipeline=Pipeline(name=f"arxiv_{config_id}", description="Arxiv data selection config", version="0.0.1"),
        model=ModelConfig(id="ArticleNet", config={"num_classes": 172}),
        model_storage=PipelineModelStorageConfig(full_model_strategy=FullModelStrategy(name="PyTorchFullModel")),
        training=TrainingConfig(
            gpus=1,
            device=gpu_device,
            dataloader_workers=1,
            use_previous_model=True,
            initial_model="random",
            batch_size=128,  # TODO(MaxiBoether): Do we want to increase this? Might affect BtS.
            optimizers=[opti_conf],
            optimization_criterion=OptimizationCriterion(name="CrossEntropyLoss"),
            checkpointing=CheckpointingConfig(activated=False),
            lr_scheduler=lr_scheduler,
            epochs_per_trigger=num_epochs,
            shuffle=True,
            amp=False,
            seed=seed,
        ),
        selection_strategy=selection_strategy,
        data=DataConfig(
            dataset_id="arxiv",
            transformations=[],
            bytes_parser_function=(
                "import torch\n"
                "def bytes_parser_function(data: memoryview) -> torch.Tensor:\n"
                "    return str(data, 'utf8')"
            ),
            tokenizer="DistilBertTokenizerTransform",
        ),
        trigger=TimeTriggerConfig(every="1d"),
        evaluation=EvaluationConfig(
            handlers=[
                EvalHandlerConfig(
                    name="exactmatrix",
                    execution_time="after_training",
                    models="matrix",
                    datasets=["arxiv-test"],
                    strategy=SlicingEvalStrategyConfig(eval_every="1d", eval_start_from=0, eval_end_at=1400000),
                )
            ],
            after_pipeline_evaluation_workers=3,
            after_training_evaluation_workers=3,
            device=gpu_device,
            result_writers=["json"],
            datasets=[
                EvalDataConfig(
                    dataset_id=dataset,
                    bytes_parser_function=(
                        "import torch\n"
                        "def bytes_parser_function(data: memoryview) -> torch.Tensor:\n"
                        "    return str(data, 'utf8')"
                    ),
                    tokenizer="DistilBertTokenizerTransform",
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
                            config={"num_classes": 172},
                        ),
                        Metric(
                            name="Accuracy",
                            evaluation_transformer_function="",
                            config={"num_classes": 172, "topn": 2},
                        ),
                        Metric(
                            name="Accuracy",
                            evaluation_transformer_function="",
                            config={"num_classes": 172, "topn": 5},
                        ),
                        Metric(
                            name="F1Score",
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.argmax(model_output, dim=-1)"
                            ),
                            config={"num_classes": 172, "average": "weighted"},
                        ),
                        Metric(
                            name="F1Score",
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.argmax(model_output, dim=-1)"
                            ),
                            config={"num_classes": 172, "average": "macro"},
                        ),
                        Metric(
                            name="F1Score",
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.argmax(model_output, dim=-1)"
                            ),
                            config={"num_classes": 172, "average": "micro"},
                        ),
                    ],
                )
                for dataset in ["arxiv", "arxiv-test"]
            ],
        ),
    )
