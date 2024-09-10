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
from modyn.config.schema.pipeline.evaluation.metrics import (
    AccuracyMetricConfig,
    F1ScoreMetricConfig,
    RocAucMetricConfig,
)
from modyn.config.schema.pipeline.model_storage import FullModelStrategy
from modyn.config.schema.pipeline.sampling.config import NewDataStrategyConfig

yb_bytes_parser_function = (
    "import torch\n"
    "import numpy as np\n"
    "def bytes_parser_function(data: bytes) -> torch.Tensor:\n"
    "    return torch.from_numpy(np.frombuffer(data, dtype=np.float32)).reshape(3, 32, 32)\n"
)
yb_evaluation_transformer_function = (
    "import torch\n"
    "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
    "    return torch.argmax(model_output, dim=-1)\n"
)
yb_evaluation_transformer_function_rocauc = (
    "import torch\n"
    "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
    "    prob_output = torch.nn.functional.softmax(model_output, dim=-1)\n"
    "    return model_output[:, 1]\n"
)


def gen_pipeline_config(
    config_ref: str,
    trigger_config: TriggerConfig,
    eval_handlers: list[EvalHandlerConfig],
    gpu_device: str,
    seed: int,
) -> ModynPipelineConfig:
    return ModynPipelineConfig(
        pipeline=Pipeline(
            name=f"yearbook_{config_ref}",
            description="Yearbook pipeline for comparing trigger policies",
            version="0.0.1",
        ),
        model=ModelConfig(id="YearbookNet", config={"num_input_channels": 3, "num_classes": 2}),
        model_storage=PipelineModelStorageConfig(full_model_strategy=FullModelStrategy(name="PyTorchFullModel")),
        training=TrainingConfig(
            gpus=1,
            device=gpu_device,
            dataloader_workers=1,
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
            epochs_per_trigger=5,
            amp=False,
            seed=seed,
        ),
        selection_strategy=NewDataStrategyConfig(
            maximum_keys_in_memory=100000, storage_backend="database", limit=-1, tail_triggers=0
        ),
        data=DataConfig(
            dataset_id="yearbook_train",
            transformations=[],
            bytes_parser_function=yb_bytes_parser_function,
        ),
        trigger=trigger_config,
        evaluation=EvaluationConfig(
            handlers=eval_handlers,
            device=gpu_device,
            after_pipeline_evaluation_workers=12,
            after_training_evaluation_workers=12,
            datasets=[
                EvalDataConfig(
                    dataset_id=yb_dataset_name,
                    bytes_parser_function=yb_bytes_parser_function,
                    batch_size=512,
                    dataloader_workers=1,
                    metrics=[
                        AccuracyMetricConfig(evaluation_transformer_function=yb_evaluation_transformer_function),
                        F1ScoreMetricConfig(
                            evaluation_transformer_function=yb_evaluation_transformer_function,
                            num_classes=2,
                            average="weighted",
                        ),
                        F1ScoreMetricConfig(
                            evaluation_transformer_function=yb_evaluation_transformer_function,
                            num_classes=2,
                            average="macro",
                        ),
                        F1ScoreMetricConfig(
                            evaluation_transformer_function=yb_evaluation_transformer_function,
                            num_classes=2,
                            average="micro",
                        ),
                        RocAucMetricConfig(evaluation_transformer_function=yb_evaluation_transformer_function_rocauc),
                    ],
                )
                for yb_dataset_name in ["yearbook_all", "yearbook_train", "yearbook_test"]
            ],
        ),
    )
