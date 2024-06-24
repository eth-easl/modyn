from modyn.config import (
    CheckpointingConfig,
    DataConfig,
    OptimizationCriterion,
    OptimizerConfig,
    OptimizerParamGroup,
    TrainingConfig,
)
from modyn.config.schema.pipeline.config import (
    EvaluationConfig,
    ModelConfig,
    ModynPipelineConfig,
    Pipeline,
    PipelineModelStorageConfig,
    TriggerConfig,
)
from modyn.config.schema.pipeline.evaluation.config import EvalDataConfig
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig
from modyn.config.schema.pipeline.evaluation.metrics import AccuracyMetricConfig, F1ScoreMetricConfig
from modyn.config.schema.pipeline.model_storage import FullModelStrategy
from modyn.config.schema.pipeline.sampling.config import NewDataStrategyConfig


def gen_pipeline_config(
    name: str, trigger: TriggerConfig, eval_handlers: list[EvalHandlerConfig]
) -> ModynPipelineConfig:
    num_classes = 2
    return ModynPipelineConfig(
        pipeline=Pipeline(name=name, description="Yearbook pipeline for comparing trigger policies", version="0.0.1"),
        model=ModelConfig(id="YearbookNet", config={"num_input_channels": 3, "num_classes": num_classes}),
        model_storage=PipelineModelStorageConfig(full_model_strategy=FullModelStrategy(name="PyTorchFullModel")),
        training=TrainingConfig(
            gpus=1,
            device="cuda:0",
            dataloader_workers=2,
            use_previous_model=True,
            initial_model="random",
            batch_size=64,
            shuffle=True,
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
        ),
        selection_strategy=NewDataStrategyConfig(
            maximum_keys_in_memory=1000, storage_backend="database", limit=-1, tail_triggers=0
        ),
        data=DataConfig(
            dataset_id="yearbook",
            transformations=[],
            bytes_parser_function=(
                "import warnings\n"
                "import torch\n"
                "def bytes_parser_function(data: memoryview) -> torch.Tensor:\n"
                "    with warnings.catch_warnings():\n"
                "        warnings.simplefilter('ignore', category=UserWarning)\n"
                "        return torch.frombuffer(data, dtype=torch.float32).reshape(3, 32, 32)"
            ),
        ),
        trigger=trigger,
        evaluation=EvaluationConfig(
            handlers=eval_handlers,
            device="cuda:0",
            result_writers=["json"],
            datasets=[
                EvalDataConfig(
                    dataset_id=yb_dataset_name,
                    bytes_parser_function=(
                        "import torch\n"
                        "import numpy as np\n"
                        "def bytes_parser_function(data: bytes) -> torch.Tensor:\n"
                        "    return torch.from_numpy(np.frombuffer(data, dtype=np.float32)).reshape(3, 32, 32)\n"
                    ),
                    batch_size=64,
                    dataloader_workers=2,
                    metrics=[
                        AccuracyMetricConfig(
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "    return torch.argmax(model_output, dim=-1)\n"
                            ),
                        ),
                        F1ScoreMetricConfig(
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.argmax(model_output, dim=-1)"
                            ),
                            num_classes=num_classes,
                            average="weighted",
                        ),
                    ],
                )
                for yb_dataset_name in ["yearbook", "yearbook_test"]
            ],
        ),
    )
