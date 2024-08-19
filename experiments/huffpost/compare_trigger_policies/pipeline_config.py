from modyn.config.schema.pipeline import (
    CheckpointingConfig,
    DataConfig,
    EvalDataConfig,
    EvalHandlerConfig,
    EvaluationConfig,
    FullModelStrategy,
    ModelConfig,
    ModynPipelineConfig,
    NewDataStrategyConfig,
    OptimizationCriterion,
    OptimizerConfig,
    OptimizerParamGroup,
    Pipeline,
    PipelineModelStorageConfig,
    TrainingConfig,
    TriggerConfig,
)
from modyn.config.schema.pipeline.evaluation.metrics import AccuracyMetricConfig, F1ScoreMetricConfig


def gen_pipeline_config(
    name: str, trigger: TriggerConfig, eval_handlers: list[EvalHandlerConfig]
) -> ModynPipelineConfig:
    num_classes = 42
    return ModynPipelineConfig(
        pipeline=Pipeline(name=name, description="Huffpost pipeline for comparing trigger policies", version="0.0.1"),
        model=ModelConfig(id="ArticleNet", config={"num_classes": num_classes}),
        model_storage=PipelineModelStorageConfig(full_model_strategy=FullModelStrategy(name="PyTorchFullModel")),
        training=TrainingConfig(
            gpus=1,
            device="cuda:0",
            dataloader_workers=8,
            use_previous_model=True,
            initial_model="random",
            batch_size=64,
            shuffle=True,
            optimizers=[
                OptimizerConfig(
                    name="default",
                    algorithm="AdamW",
                    source="PyTorch",
                    param_groups=[
                        OptimizerParamGroup(
                            module="model", config={"lr": 0.00005, "momentum": 0.9, "weight_decay": 0.01}
                        )
                    ],
                )
            ],
            optimization_criterion=OptimizationCriterion(name="CrossEntropyLoss"),
            checkpointing=CheckpointingConfig(activated=False),
            seed=42,
            epochs_per_trigger=5,
        ),
        selection_strategy=NewDataStrategyConfig(
            maximum_keys_in_memory=100000, storage_backend="database", limit=-1, tail_triggers=0
        ),
        data=DataConfig(
            dataset_id="huffpost_kaggle_train",
            bytes_parser_function=("def bytes_parser_function(data: bytes) -> str:\n" "    return str(data, 'utf8')"),
            tokenizer="DistilBertTokenizerTransform",
        ),
        trigger=trigger,
        evaluation=EvaluationConfig(
            handlers=eval_handlers,
            device="cuda:0",
            after_training_evaluation_workers=10,
            after_pipeline_evaluation_workers=10,
            datasets=[
                EvalDataConfig(
                    dataset_id=hp_dataset_name,
                    bytes_parser_function=(
                        "def bytes_parser_function(data: bytes) -> str:\n" "    return str(data, 'utf8')"
                    ),
                    tokenizer="DistilBertTokenizerTransform",
                    batch_size=64,
                    dataloader_workers=8,
                    metrics=[
                        AccuracyMetricConfig(
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "    return torch.argmax(model_output, dim=-1)"
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
                for hp_dataset_name in ["huffpost_kaggle", "huffpost_kaggle_test"]
            ],
        ),
    )
