from modyn.config.schema.pipeline import (
    CheckpointingConfig,
    DataConfig,
    EvalDataConfig,
    EvalHandlerConfig,
    EvaluationConfig,
    FullModelStrategy,
    Metric,
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


def gen_pipeline_config(
    name: str, trigger: TriggerConfig, eval_handlers: list[EvalHandlerConfig]
) -> ModynPipelineConfig:
    return ModynPipelineConfig(
        pipeline=Pipeline(name=name, description="Huffpost pipeline for comparing trigger policies", version="0.0.1"),
        model=ModelConfig(id="ArticleNet", config={"num_classes": 55}),
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
                    algorithm="SGD",
                    source="PyTorch",
                    param_groups=[
                        OptimizerParamGroup(
                            module="model", config={"lr": 0.00002, "momentum": 0.9, "weight_decay": 0.01}
                        )
                    ],
                )
            ],
            optimization_criterion=OptimizationCriterion(name="CrossEntropyLoss"),
            checkpointing=CheckpointingConfig(activated=False),
            seed=42,
            epochs_per_trigger=1,
        ),
        selection_strategy=NewDataStrategyConfig(
            maximum_keys_in_memory=1000, storage_backend="database", limit=-1, tail_triggers=0
        ),
        data=DataConfig(
            dataset_id="huffpost",
            bytes_parser_function=("def bytes_parser_function(data: bytes) -> str:\n" "    return str(data, 'utf8')"),
            tokenizer="DistilBertTokenizerTransform",
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
                        "def bytes_parser_function(data: bytes) -> str:\n" "    return str(data, 'utf8')"
                    ),
                    tokenizer="DistilBertTokenizerTransform",
                    batch_size=64,
                    dataloader_workers=8,
                    metrics=[
                        Metric(
                            name="Accuracy",
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "    return torch.argmax(model_output, dim=-1)"
                            ),
                            config={"num_classes": 2, "average": "weighted"},
                        ),
                        Metric(
                            name="F1score",
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.argmax(model_output, dim=-1)"
                            ),
                            config={"num_classes": 2, "average": "weighted"},
                        ),
                    ],
                )
                for yb_dataset_name in ["huffpost_kaggle", "huffpost_kaggle_test"]
            ],
        ),
    )
