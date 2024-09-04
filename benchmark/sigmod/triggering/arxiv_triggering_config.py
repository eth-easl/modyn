from __future__ import annotations

from modyn.config import (
    CheckpointingConfig,
    OptimizationCriterion,
    OptimizerConfig,
    OptimizerParamGroup,
)
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
from modyn.utils.utils import SECONDS_PER_UNIT


def gen_arxiv_training_conf(gpu_device: str, seed: int):
    opti_conf = OptimizerConfig(
        name="default",
        algorithm="AdamW",
        source="PyTorch",
        param_groups=[OptimizerParamGroup(module="model", config={"lr": 0.00002, "weight_decay": 0.01})],
    )

    return TrainingConfig(
        gpus=1,
        device=gpu_device,
        dataloader_workers=1,
        use_previous_model=True,
        initial_model="random",
        batch_size=128,
        optimizers=[opti_conf],
        optimization_criterion=OptimizationCriterion(name="CrossEntropyLoss"),
        checkpointing=CheckpointingConfig(activated=False),
        lr_scheduler=None,
        epochs_per_trigger=5,
        shuffle=True,
        amp=False,
        seed=seed,
    )


ARXIV_BPF = (
    "import torch\n" "def bytes_parser_function(data: memoryview) -> torch.Tensor:\n" "    return str(data, 'utf8')"
)

ARXIV_EVAL_FUNC = (
    "import torch\n"
    "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n"
    "    return torch.argmax(model_output, dim=-1)"
)


def get_eval_data_config(dataset: str) -> EvalDataConfig:
    return EvalDataConfig(
        dataset_id=dataset,
        bytes_parser_function=ARXIV_BPF,
        tokenizer="DistilBertTokenizerTransform",
        batch_size=256,
        dataloader_workers=1,
        metrics=[
            AccuracyMetricConfig(
                evaluation_transformer_function=ARXIV_EVAL_FUNC,
                topn=1,
            ),
            AccuracyMetricConfig(evaluation_transformer_function="", topn=2),
            AccuracyMetricConfig(evaluation_transformer_function="", topn=5),
            F1ScoreMetricConfig(
                evaluation_transformer_function=ARXIV_EVAL_FUNC,
                num_classes=172,
                average="weighted",
            ),
            F1ScoreMetricConfig(
                evaluation_transformer_function=ARXIV_EVAL_FUNC,
                num_classes=172,
                average="macro",
            ),
            F1ScoreMetricConfig(
                evaluation_transformer_function=ARXIV_EVAL_FUNC,
                num_classes=172,
                average="micro",
            ),
        ],
    )


def gen_arxiv_triggering_config(
    config_id: str, gpu_device: str, trigger_config: TriggerConfig, seed: int, start_eval_at: int
) -> ModynPipelineConfig:
    return ModynPipelineConfig(
        pipeline=Pipeline(name=f"kaggle_arxiv_{config_id}", description="Arxiv triggering config", version="0.0.1"),
        model=ModelConfig(id="ArticleNet", config={"num_classes": 172}),
        model_storage=PipelineModelStorageConfig(full_model_strategy=FullModelStrategy(name="PyTorchFullModel")),
        training=gen_arxiv_training_conf(gpu_device, seed),
        selection_strategy=NewDataStrategyConfig(
            maximum_keys_in_memory=200000, storage_backend="database", tail_triggers=0, limit=-1
        ),
        data=DataConfig(
            dataset_id="arxiv_kaggle_train",
            bytes_parser_function=ARXIV_BPF,
            tokenizer="DistilBertTokenizerTransform",
        ),
        trigger=trigger_config,
        evaluation=EvaluationConfig(
            handlers=[
                EvalHandlerConfig(
                    name="exactmatrix",
                    execution_time="after_pipeline",
                    models="matrix",
                    datasets=["arxiv_kaggle_test"],
                    strategy=PeriodicEvalStrategyConfig(
                        every="26w",
                        interval="[-13w; +13w]",
                        start_timestamp=start_eval_at + 13 * SECONDS_PER_UNIT["w"],
                        end_timestamp=1724803200,
                    ),
                )
            ],
            after_pipeline_evaluation_workers=2,
            after_training_evaluation_workers=2,
            device=gpu_device,
            datasets=[get_eval_data_config(dataset) for dataset in ["arxiv_kaggle_test"]],
        ),
    )
