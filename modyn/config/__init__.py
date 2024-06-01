from pathlib import Path

import yaml
from modyn.config.schema.pipeline_component.sampling.downsampling_config import (  # noqa: F401
    MultiDownsamplingConfig,
    SingleDownsamplingConfig,
)

from .schema.config import (
    BinaryFileByteOrder,
    DatabaseConfig,
    DatabaseMixin,
    DatasetBinaryFileWrapperConfig,
    DatasetCsvFileWrapperConfig,
    DatasetFileWrapperConfig,
    DatasetPngFileWrapperConfig,
    DatasetsConfig,
    EvaluatorConfig,
    HostnamePortMixin,
    MetadataDatabaseConfig,
    MetadataProcessorConfig,
    ModelStorageConfig,
    ModynConfig,
    ProjectConfig,
    SelectorConfig,
    StorageConfig,
    SupervisorConfig,
    TensorboardConfig,
    TrainingServerConfig,
)
from modyn.config.schema.pipeline import (  # noqa: F401
    CoresetStrategyConfig,
    DataConfig,
    EvalDataConfig,
    EvaluationConfig,
    FreshnessSamplingStrategyConfig,
    FullModelStrategy,
    IncrementalModelStrategy,
    Metric,
    ModelConfig,
    ModynPipelineConfig,
    NewDataStrategyConfig,
    Pipeline,
    PipelineModelStorageConfig,
    PresamplingConfig,
    ResultWriter,
    SelectionStrategy,
    StorageBackend,
    TrainingConfig,
    TriggerConfig,
)

__all__ = [
    # Modyn config models
    "HostnamePortMixin",
    "DatabaseMixin",
    "ProjectConfig",
    "BinaryFileByteOrder",
    "DatasetCsvFileWrapperConfig",
    "DatasetBinaryFileWrapperConfig",
    "DatasetPngFileWrapperConfig",
    "DatasetFileWrapperConfig",
    "DatasetsConfig",
    "DatabaseConfig",
    "StorageConfig",
    "ModelStorageConfig",
    "EvaluatorConfig",
    "MetadataDatabaseConfig",
    "SelectorConfig",
    "TrainingServerConfig",
    "MetadataProcessorConfig",
    "TensorboardConfig",
    "SupervisorConfig",
    "ModynConfig",
    # Pipeline config models
    "Pipeline",
    "ModelConfig",
    "FullModelStrategy",
    "IncrementalModelStrategy",
    "PipelineModelStorageConfig",
    "PresamplingConfig",
    "SingleDownsamplingConfig",
    "FreshnessSamplingStrategyConfig",
    "SelectionStrategy",
    "TrainingConfig",
    "DataConfig",
    "TriggerConfig",
    "Metric",
    "EvalDataConfig",
    "ResultWriter",
    "EvaluationConfig",
    "ModynPipelineConfig",
    "StorageBackend",
    # file readers
    "read_modyn_config",
    "read_pipeline",
]


def read_modyn_config(path: Path) -> ModynConfig:
    """Read a Modyn configuration file and validate it.

    Args:
        path: Path to the configuration file.

    Returns:
        The validated Modyn config as a pydantic model.
    """
    with open(path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return ModynConfig.model_validate(config)


def read_pipeline(path: Path) -> ModynPipelineConfig:
    """Read a pipeline configuration file.

    Args:
        path: Path to the pipeline configuration file.

    Returns:
        The validated pipeline configuration as a pydantic model.
    """
    with open(path, "r", encoding="utf-8") as pipeline_file:
        pipeline = yaml.safe_load(pipeline_file)
    return ModynPipelineConfig.model_validate(pipeline)
