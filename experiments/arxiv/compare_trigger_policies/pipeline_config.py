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

arxiv_bytes_parser_function = (
    "import torch\n"
    "import numpy as np\n"
    "def bytes_parser_function(data: bytes) -> str:\n"
    "    return str(data, 'utf8')"
)
arxiv_evaluation_transformer_function = (
    "import torch\n"
    "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
    "    return torch.argmax(model_output, dim=-1)\n"
)


def gen_pipeline_config(
    config_ref: str,
    trigger_config: TriggerConfig,
    eval_handlers: list[EvalHandlerConfig],
    gpu_device: str,
    seed: int,
) -> ModynPipelineConfig:
    num_classes = 172
    return ModynPipelineConfig(
        pipeline=Pipeline(
            name=config_ref, description="Arxiv pipeline for comparing trigger policies", version="0.0.1"
        ),
        model=ModelConfig(id="ArticleNet", config={"num_classes": num_classes}),
        model_storage=PipelineModelStorageConfig(full_model_strategy=FullModelStrategy(name="PyTorchFullModel")),
        training=TrainingConfig(
            gpus=1,
            device=gpu_device,
            dataloader_workers=1,
            use_previous_model=True,
            initial_model="random",
            batch_size=128,  # gpu memory limit does't allow for larger batch sizes
            shuffle=True,
            optimizers=[
                OptimizerConfig(
                    name="default",
                    algorithm="AdamW",
                    source="PyTorch",
                    param_groups=[OptimizerParamGroup(module="model", config={"lr": 0.00002, "weight_decay": 0.01})],
                )
            ],
            optimization_criterion=OptimizationCriterion(name="CrossEntropyLoss"),
            checkpointing=CheckpointingConfig(activated=False),
            epochs_per_trigger=5,
            amp=False,
            seed=seed,
        ),
        selection_strategy=NewDataStrategyConfig(
            maximum_keys_in_memory=200000, storage_backend="database", limit=-1, tail_triggers=0
        ),
        data=DataConfig(
            dataset_id="arxiv_kaggle_train",
            bytes_parser_function=arxiv_bytes_parser_function,
            tokenizer="DistilBertTokenizerTransform",
        ),
        trigger=trigger_config,
        evaluation=EvaluationConfig(
            handlers=eval_handlers,
            device=gpu_device,
            after_pipeline_evaluation_workers=8,
            after_training_evaluation_workers=8,
            datasets=[
                EvalDataConfig(
                    dataset_id=yb_dataset_name,
                    bytes_parser_function=arxiv_bytes_parser_function,
                    batch_size=512,
                    dataloader_workers=1,
                    tokenizer="DistilBertTokenizerTransform",
                    metrics=[
                        AccuracyMetricConfig(
                            evaluation_transformer_function=arxiv_evaluation_transformer_function, topn=1
                        ),
                        AccuracyMetricConfig(evaluation_transformer_function="", topn=2),
                        AccuracyMetricConfig(evaluation_transformer_function="", topn=5),
                        AccuracyMetricConfig(evaluation_transformer_function="", topn=10),
                        F1ScoreMetricConfig(
                            evaluation_transformer_function=arxiv_evaluation_transformer_function,
                            num_classes=num_classes,
                            average="weighted",
                        ),
                        F1ScoreMetricConfig(
                            evaluation_transformer_function=arxiv_evaluation_transformer_function,
                            num_classes=num_classes,
                            average="macro",
                        ),
                        F1ScoreMetricConfig(
                            evaluation_transformer_function=arxiv_evaluation_transformer_function,
                            num_classes=num_classes,
                            average="micro",
                        ),
                        # RocAucMetric is traditionally used for binary classification
                    ],
                )
                for yb_dataset_name in ["arxiv_kaggle_all", "arxiv_kaggle_train", "arxiv_kaggle_test"]
            ],
        ),
    )
