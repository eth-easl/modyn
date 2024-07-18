from modyn.config import LrSchedulerConfig, TrainingConfig, OptimizerConfig, OptimizerParamGroup, OptimizationCriterion, \
    CheckpointingConfig, SelectionStrategy, ModynPipelineConfig, ModelConfig, Pipeline, PipelineModelStorageConfig, \
    FullModelStrategy, DataConfig, DataAmountTriggerConfig, EvaluationConfig, EvalHandlerConfig, \
    StaticEvalStrategyConfig, EvalDataConfig, AccuracyMetricConfig, F1ScoreMetricConfig


def gen_cifar10_training_conf(
        optimizer: str, lr: float, gpu_device: str, lr_scheduler: LrSchedulerConfig | None, num_epochs: int, seed: int
):
    assert optimizer == "AdamW"
    del lr  # hardcode
    return TrainingConfig(
        gpus=1,
        device=gpu_device,
        dataloader_workers=1,
        use_previous_model=True,
        initial_model="random",
        batch_size=320,  # TODO(MaxiBoether): Do we want to increase this? Might affect BtS.
        optimizers=[
            OptimizerConfig(
                name="default",
                algorithm=optimizer,
                source="PyTorch",
                param_groups=[OptimizerParamGroup(module="model")],
            )
        ],
        optimization_criterion=OptimizationCriterion(name="CrossEntropyLoss"),
        checkpointing=CheckpointingConfig(activated=False),
        lr_scheduler=lr_scheduler,
        epochs_per_trigger=num_epochs,
        shuffle=True,
        amp=False,
        seed=seed,
        record_loss_every=1,
    )


def gen_cifar10_config(
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
        trigger_period: str,
) -> ModynPipelineConfig:
    del model  # hardcore resnet50
    del lr
    del optimizer
    del dataset
    model_config = ModelConfig(id="ResNet18", config={"num_classes": num_classes})

    bytes_parser_func = (
        "from PIL import Image\n"
        "import io\n"
        "def bytes_parser_function(data: memoryview) -> Image:\n"
        "   return Image.open(io.BytesIO(data)).convert('RGB')"
    )

    transformations = [
        "transforms.ToTensor()",
        "transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))",
    ]

    return ModynPipelineConfig(
        pipeline=Pipeline(name=f"cifar10_{config_id}", description="CIFAR10 data selection config", version="0.0.1"),
        model=model_config,
        model_storage=PipelineModelStorageConfig(full_model_strategy=FullModelStrategy(name="PyTorchFullModel")),
        training=gen_cifar10_training_conf("AdamW", 0.42, gpu_device, lr_scheduler, num_epochs, seed),
        selection_strategy=selection_strategy,
        data=DataConfig(dataset_id="cifar10-train", transformations=transformations, bytes_parser_function=bytes_parser_func),
        trigger=DataAmountTriggerConfig(num_samples=50000),
        evaluation=EvaluationConfig(
            handlers=[
                EvalHandlerConfig(
                    name="default_eval",
                    execution_time="after_pipeline",
                    models="matrix",
                    datasets=[f"cifar10-test"],
                    strategy=StaticEvalStrategyConfig(
                        intervals=[(0, None)],
                    ),
                )
            ],
            after_pipeline_evaluation_workers=4,
            after_training_evaluation_workers=4,
            device=gpu_device,
            result_writers=["json"],
            datasets=[
                EvalDataConfig(
                    dataset_id="cifar10-test",
                    bytes_parser_function=bytes_parser_func,
                    transformations=transformations,
                    batch_size=256,
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
                        AccuracyMetricConfig(evaluation_transformer_function="", topn=2),
                        AccuracyMetricConfig(evaluation_transformer_function="", topn=5),
                        F1ScoreMetricConfig(
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.argmax(model_output, dim=-1)"
                            ),
                            num_classes=num_classes,
                            average="weighted",
                        ),
                        F1ScoreMetricConfig(
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.argmax(model_output, dim=-1)"
                            ),
                            num_classes=num_classes,
                            average="macro",
                        ),
                        F1ScoreMetricConfig(
                            evaluation_transformer_function=(
                                "import torch\n"
                                "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
                                "   return torch.argmax(model_output, dim=-1)"
                            ),
                            num_classes=num_classes,
                            average="micro",
                        ),
                    ],
                )
            ],
        ),
    )
