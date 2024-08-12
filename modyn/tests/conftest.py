# pylint: disable=redefined-outer-name

import pytest

from modyn.config.schema.pipeline import (
    AccuracyMetricConfig,
    CheckpointingConfig,
    DataAmountTriggerConfig,
    DataConfig,
    EvalDataConfig,
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
)
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig
from modyn.config.schema.pipeline.evaluation.strategy.slicing import SlicingEvalStrategyConfig
from modyn.config.schema.system.config import (
    DatabaseConfig,
    DatasetBinaryFileWrapperConfig,
    DatasetsConfig,
    EvaluatorConfig,
    MetadataDatabaseConfig,
    ModelStorageConfig,
    ModynConfig,
    SelectorConfig,
    StorageConfig,
    TrainingServerConfig,
)

# --------------------------------------------------- Modyn config --------------------------------------------------- #


@pytest.fixture
def storage_config() -> StorageConfig:
    return StorageConfig(
        port=50051,
        hostname="storage",
        sample_batch_size=2000000,
        sample_dbinsertion_batchsize=1000000,
        insertion_threads=8,
        retrieval_threads=8,
        sample_table_unlogged=True,
        file_watcher_watchdog_sleep_time_s=5,
        datasets=[],
        database=DatabaseConfig(
            drivername="postgresql",
            username="postgres",
            password="postgres",
            hostname="storage-db",
            port="5432",
            database="postgres",
        ),
    )


@pytest.fixture
def model_storage_config() -> ModelStorageConfig:
    return ModelStorageConfig(
        hostname="model_storage",
        port="50059",
        ftp_port="50060",
        models_directory="/tmp/models",
    )


@pytest.fixture
def evaluator_config() -> EvaluatorConfig:
    return EvaluatorConfig(hostname="evaluator", port="50061")


@pytest.fixture
def metadata_db_config() -> MetadataDatabaseConfig:
    return MetadataDatabaseConfig(
        drivername="postgresql",
        username="postgres",
        password="postgres",
        hostname="metadata-db",
        port="5432",
        database="postgres",
        hash_partition_modulus=32,
    )


@pytest.fixture
def selector_config() -> SelectorConfig:
    return SelectorConfig(
        hostname="selector",
        port="50056",
        keys_in_selector_cache=500000,
        sample_batch_size=500000,
        insertion_threads=16,
        trigger_sample_directory="/tmp/trigger_samples",
        local_storage_directory="/tmp/local_storage",
        cleanup_storage_directories_after_shutdown=True,
        ignore_existing_trigger_samples=False,
    )


@pytest.fixture
def trainer_server_config() -> TrainingServerConfig:
    return TrainingServerConfig(
        hostname="trainer_server",
        port="50057",
        ftp_port="50058",
        offline_dataset_directory="/tmp/offline_dataset",
    )


@pytest.fixture
def dummy_system_config(
    storage_config: StorageConfig,
    model_storage_config: ModelStorageConfig,
    evaluator_config: EvaluatorConfig,
    metadata_db_config: MetadataDatabaseConfig,
    selector_config: SelectorConfig,
    trainer_server_config: TrainingServerConfig,
) -> ModynConfig:
    return ModynConfig(
        storage=storage_config,
        model_storage=model_storage_config,
        evaluator=evaluator_config,
        metadata_database=metadata_db_config,
        selector=selector_config,
        trainer_server=trainer_server_config,
    )


@pytest.fixture
def dummy_dataset_config() -> DatasetsConfig:
    return DatasetsConfig(
        name="test",
        description="",
        version="",
        base_path="",
        filesystem_wrapper_type="BinaryFilesystemWrapper",
        file_wrapper_type="BinaryFileWrapper",
        file_wrapper_config=DatasetBinaryFileWrapperConfig(
            file_extension=".bin",
            byteorder="little",
            record_size=8,
            label_size=4,
        ),
        selector_batch_size=128,
    )


# ----------------------------------------------------- Pipeline ----------------------------------------------------- #


@pytest.fixture
def pipeline_training_config() -> TrainingConfig:
    return TrainingConfig(
        gpus=1,
        device="cpu",
        dataloader_workers=1,
        use_previous_model=True,
        initial_model="random",
        batch_size=42,
        optimizers=[
            OptimizerConfig(
                name="default1", algorithm="SGD", source="PyTorch", param_groups=[OptimizerParamGroup(module="model")]
            )
        ],
        optimization_criterion=OptimizationCriterion(name="CrossEntropyLoss"),
        checkpointing=CheckpointingConfig(activated=False),
        shuffle=False,
    )


@pytest.fixture
def eval_strategies_config() -> SlicingEvalStrategyConfig:
    return SlicingEvalStrategyConfig(eval_every="100s", eval_start_from=0, eval_end_at=300)


@pytest.fixture
def pipeline_evaluation_config(eval_strategies_config: SlicingEvalStrategyConfig) -> EvaluationConfig:
    return EvaluationConfig(
        handlers=[
            EvalHandlerConfig(
                execution_time="after_training",
                strategy=eval_strategies_config,
                models="matrix",
                datasets=["MNIST_eval"],
            )
        ],
        device="cpu",
        datasets=[
            EvalDataConfig(
                dataset_id="MNIST_eval",
                bytes_parser_function="def bytes_parser_function(data: bytes) -> bytes:\n\treturn data",
                dataloader_workers=2,
                batch_size=64,
                metrics=[AccuracyMetricConfig()],
            )
        ],
    )


@pytest.fixture
def dummy_pipeline_config(pipeline_training_config: TrainingConfig) -> ModynPipelineConfig:
    return ModynPipelineConfig(
        pipeline=Pipeline(name="Test"),
        model=ModelConfig(id="ResNet18"),
        model_storage=PipelineModelStorageConfig(full_model_strategy=FullModelStrategy(name="PyTorchFullModel")),
        training=pipeline_training_config,
        data=DataConfig(
            dataset_id="test",
            bytes_parser_function="def bytes_parser_function(x):\n\treturn x",
        ),
        trigger=DataAmountTriggerConfig(num_samples=1),
        selection_strategy=NewDataStrategyConfig(
            maximum_keys_in_memory=10,
            tail_triggers=None,
        ),
    )
