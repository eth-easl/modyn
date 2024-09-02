from __future__ import annotations

from modyn.config import CheckpointingConfig, OptimizationCriterion, OptimizerConfig, OptimizerParamGroup
from modyn.config.schema.pipeline import (
    AccuracyMetricConfig,
    DataConfig,
    EvalDataConfig,
    EvaluationConfig,
    F1ScoreMetricConfig,
    FullModelStrategy,
    ModelConfig,
    ModynPipelineConfig,
    Pipeline,
    PipelineModelStorageConfig,
    TrainingConfig,
)
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig
from modyn.config.schema.pipeline.evaluation.strategy.periodic import PeriodicEvalStrategyConfig
from modyn.config.schema.pipeline.sampling.config import NewDataStrategyConfig
from modyn.config.schema.pipeline.trigger import TriggerConfig

YEARBOOK_BYTES_PARSER_FUNC = (
    "import warnings\n"
    "import torch\n"
    "def bytes_parser_function(data: memoryview) -> torch.Tensor:\n"
    "    with warnings.catch_warnings():\n"
    "       warnings.simplefilter('ignore', category=UserWarning)\n"
    "       return torch.frombuffer(data, dtype=torch.float32).reshape(3, 32, 32)"
)


def get_eval_data_config(dataset: str) -> EvalDataConfig:
    return EvalDataConfig(
        dataset_id=dataset,
        bytes_parser_function=YEARBOOK_BYTES_PARSER_FUNC,
        batch_size=512,
        dataloader_workers=1,
        metrics=[
            AccuracyMetricConfig(
                evaluation_transformer_function=(
                    "import torch\n"
                    "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                    "    return torch.argmax(model_output, dim=-1)"
                ),
                topn=1,
            ),
            F1ScoreMetricConfig(
                evaluation_transformer_function=(
                    "import torch\n"
                    "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                    "   return torch.argmax(model_output, dim=-1)"
                ),
                num_classes=2,
                average="weighted",
            ),
            F1ScoreMetricConfig(
                evaluation_transformer_function=(
                    "import torch\n"
                    "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                    "   return torch.argmax(model_output, dim=-1)"
                ),
                num_classes=2,
                average="macro",
            ),
            F1ScoreMetricConfig(
                evaluation_transformer_function=(
                    "import torch\n"
                    "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                    "   return torch.argmax(model_output, dim=-1)"
                ),
                num_classes=2,
                average="micro",
            ),
        ],
    )


def gen_yearbook_triggering_config(
    config_id: str,
    gpu_device: str,
    trigger_config: TriggerConfig,
    seed: int,
) -> ModynPipelineConfig:
    return ModynPipelineConfig(
        pipeline=Pipeline(name=f"yearbook_{config_id}", description="Yearbook triggering config", version="0.0.1"),
        model=ModelConfig(id="YearbookNet", config={"num_input_channels": 3, "num_classes": 2}),
        model_storage=PipelineModelStorageConfig(full_model_strategy=FullModelStrategy(name="PyTorchFullModel")),
        training=TrainingConfig(
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
            epochs_per_trigger=5,
            shuffle=True,
            amp=False,
            seed=seed,
        ),
        selection_strategy=NewDataStrategyConfig(
            maximum_keys_in_memory=100000, storage_backend="database", tail_triggers=0, limit=-1
        ),
        data=DataConfig(dataset_id="yearbook_train", bytes_parser_function=YEARBOOK_BYTES_PARSER_FUNC),
        trigger=trigger_config,
        evaluation=EvaluationConfig(
            handlers=[
                EvalHandlerConfig(
                    name="slidingmatrix",
                    execution_time="after_pipeline",
                    models="matrix",
                    datasets=["yearbook_test"],
                    strategy=PeriodicEvalStrategyConfig(
                        every="1d", interval="[-25h; +25h]", start_timestamp=0, end_timestamp=7258000
                    ),
                ),
            ],
            after_pipeline_evaluation_workers=12,
            after_training_evaluation_workers=12,
            device=gpu_device,
            datasets=[get_eval_data_config(dataset) for dataset in ["yearbook_train", "yearbook_test"]],
        ),
    )
