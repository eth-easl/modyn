from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel, Field

# ------------------------------------------------------ MIXINS ------------------------------------------------------ #


class HostnamePortMixin(BaseModel):
    """Mixin for hostname and port configuration."""

    hostname: str = Field(description="The hostname where the service can be reached.")
    port: int = Field(description="The port where the service can be reached.")

    @property
    def address(self) -> str:
        return f"{self.hostname}:{self.port}"


class DatabaseMixin(HostnamePortMixin):
    """Mixin for a database configuration."""

    drivername: str = Field(description="The drivername to use for the database.")
    username: str = Field(description="The username to use for the database.")
    password: str = Field(description="The password to use for the database.")
    database: str = Field(description="The database to use for the database.")


# ------------------------------------------------------ PROJECT ----------------------------------------------------- #


class ProjectConfig(BaseModel):
    """Represents the configuration of a project."""

    name: str = Field("modyn", description="The name of the project.")
    description: str | None = Field(None, description="The description of the project.")
    version: str | None = Field(None, description="The version of the project.")


# ------------------------------------------------------ STORAGE ----------------------------------------------------- #

BinaryFileByteOrder = Literal["big", "little"]


class _DatasetBaseFileWrapperConfig(BaseModel):
    """Represents a dataset file used by modyn."""

    file_extension: str = Field(description="The file extension of the dataset.", pattern=r"^\..*$")


class DatasetCsvFileWrapperConfig(_DatasetBaseFileWrapperConfig):
    """Represents a csv dataset file used by modyn."""

    separator: str = Field(",", description="The separator used in CSV files.")
    quote: str = Field("\\0", description="The quote character used in CSV files.")
    quoted_linebreaks: bool = Field(True, description="Whether linebreaks are quoted in CSV files.")

    label_index: int = Field(
        -1,
        description=(
            "Column index of the label. For columns 'width, 'height, 'age', 'label' you should set label_index to 3."
        ),
    )
    target_index: int = Field(
        -1,
        description=(
            "Column index of the label. For columns 'width, 'height, 'age', 'label' you should set label_index to 3."
        ),
    )
    ignore_first_line: bool = Field(
        False, description="If the first line is the table header, you can skip it setting this parameter to True."
    )
    encoding: str = Field("utf-8", description="Encoding of the CSV files.")
    validate_file_content: bool = Field(
        True,
        description=(
            "Whether to validate the file content before inserting the data. It checks that it is a csv, that all "
            "rows are the same size and that the 'label' column exists."
        ),
    )
    has_labels: bool = Field(True, description="Describes whether the dataset contains a label field or not")
    has_targets: bool = Field(
        False, description="Describes whether the dataset contains a generative_target field or not"
    )


class DatasetBinaryFileWrapperConfig(_DatasetBaseFileWrapperConfig):
    """Represents a binary dataset file used by modyn."""

    byteorder: BinaryFileByteOrder = Field(
        description="The byteorder when reading an integer from multibyte data in a binary file"
    )
    record_size: int = Field(description="The size of each full record in bytes (label + features).")
    label_size: int = Field(description="The size of the label field in bytes for a binary file wrapper.")
    target_size: int = Field(0,description="The size of the generative targets field in bytes for a binary file wrapper.")
    has_labels: bool = Field(True, description="Describes whether the dataset contains a label field or not")
    has_targets: bool = Field(
        False, description="Describes whether the dataset contains a generative_target field or not"
    )


class DatasetPngFileWrapperConfig(_DatasetBaseFileWrapperConfig):
    """Represents a png dataset file used by modyn."""

    label_file_extension: str = Field(description="The label file extension of the dataset", pattern=r"^\..*$")
    has_labels: bool = Field(True, description="Describes whether the dataset contains a label field or not")
    has_targets: bool = Field(
        False, description="Describes whether the dataset contains a generative_target field or not"
    )


DatasetFileWrapperConfig = Union[  # noqa: UP007
    DatasetCsvFileWrapperConfig, DatasetBinaryFileWrapperConfig, DatasetPngFileWrapperConfig
]


class DatasetsConfig(BaseModel):
    """Configures a dataset to be used by modyn."""

    name: str = Field(description="The name of the dataset.")
    description: str = Field(description="The description of the dataset.")
    version: str = Field(description="The version of the dataset.")
    base_path: str = Field(description="The base path of the dataset.")
    filesystem_wrapper_type: str = Field(description="The filesystem wrapper type of the dataset.")
    file_wrapper_type: str = Field(description="The file wrapper type of the dataset.")
    file_wrapper_config: DatasetFileWrapperConfig = Field(description="The file wrapper config of the dataset.")
    ignore_last_timestamp: bool | None = Field(
        None,
        description=(
            "Whether to ignore the last timestamp when scanning for new files, i.e., if this is set to false, in "
            "case a new file gets added to the storage that has a smaller timestamp than the latest file that the "
            "storage has already processed, the file is not processed."
        ),
    )
    file_watcher_interval: int | None = Field(
        None, description="The interval in seconds in which the file watcher checks for new files."
    )
    selector_batch_size: int = Field(
        128,
        description="The number of samples per which we check for triggers and inform the selector.",
    )


class DatabaseConfig(DatabaseMixin):
    """Configuration for modyn's main database."""

    hash_partition_modulus: int | None = Field(
        None, description="The modulus to use for the hash partitioning of the samples."
    )


class StorageConfig(HostnamePortMixin):
    """Configuration for modyn's storage engine."""

    sample_batch_size: int = Field(
        10000,
        description=(
            "The size of a batch when requesting new samples from storage. All new samples are returned, however, "
            "to reduce the size of a single answer the keys are batched in sizes of `sample_batch_size`."
        ),
    )

    sample_dbinsertion_batchsize: int = Field(
        description=(
            "How many samples are at least required when scanning new files to trigger an intermediate insertion "
            "request."
        )
    )
    insertion_threads: int = Field(
        description=(
            "The number of threads used to insert samples into the storage DB. If set to <= 0, multithreaded inserts "
            "are disabled."
        )
    )
    retrieval_threads: int | None = Field(
        None,
        description=(
            "The number of threads used to get samples from the storage DB. If set to <= 1, multithreaded gets "
            "are disabled."
        ),
    )
    sample_table_unlogged: bool = Field(
        True,
        description=(
            "This configures whether the table storing all samples is UNLOGGED (= high performance) or crash "
            "resilient. For datasets with many samples (such as Criteo), this is recommended for highest insertion "
            "performance. In other scenarios, this might not be necessary."
        ),
    )

    force_fallback_insert: bool = Field(
        False, description="Enforces fallback insert functionality instead of potentially optimized techniques."
    )
    file_watcher_watchdog_sleep_time_s: int = Field(
        3,
        description=(
            "The time in seconds the file watcher watchdog sleeps between checking if the file watchers are "
            "still alive."
        ),
    )
    datasets: list[DatasetsConfig] = Field(
        default_factory=list, description="The datasets to use for the storage engine."
    )
    database: DatabaseConfig = Field(description="The database configuration.")


# --------------------------------------------------- MODEL STORAGE -------------------------------------------------- #


class ModelStorageConfig(HostnamePortMixin):
    """Configuration for modyn's model storage component and its grpc
    service."""

    ftp_port: str = Field(description="The port of the FDP server used by the model_storage component.")
    models_directory: str | None = Field(None, description="The directory where we store the trained models.")


# ----------------------------------------------------- EVALUATOR ---------------------------------------------------- #


class EvaluatorConfig(HostnamePortMixin):
    """Configuration for modyn's evaluation component."""


# ---------------------------------------------------- METADATA DB --------------------------------------------------- #


class MetadataDatabaseConfig(DatabaseConfig):
    """Configuration for modyn's metadata database."""

    hash_partition_modulus: int | None = Field(
        None, description="The modulus to use for the hash partitioning of the metadata."
    )
    seed: float | None = Field(None, description="If provided, this number is used to seed the database.", ge=-1, le=1)


# ----------------------------------------------------- SELECTOR ----------------------------------------------------- #


class SelectorConfig(HostnamePortMixin):
    """Configuration for modyn's selector."""

    keys_in_selector_cache: int = Field(description="How many keys each selector is allowed to cache in memory.")
    sample_batch_size: int = Field(
        description=(
            "The size of a batch when requesting sample keys for a trigger partition and worker. All new samples are "
            "returned, however, to reduce the size of a single answer the keys are batched in sizes "
            "of `sample_batch_size`."
        )
    )
    insertion_threads: int = Field(
        description=(
            "The number of threads used to insert samples into the metadata DB. If set to <= 0, multithreaded inserts "
            "are disabled."
        )
    )
    trigger_sample_directory: str = Field(
        description=(
            "Directory where the the TriggerTrainingSet (fixed set of samples to train on for one trigger) is stored."
        )
    )
    local_storage_directory: str = Field(
        description="The directory where selection strategies that use the local storage backend persist data to."
    )
    local_storage_max_samples_in_file: int = Field(
        1000000,
        description=("The maximum amount of samples to be persisted in a single file for the local storage backend."),
    )
    cleanup_storage_directories_after_shutdown: bool = Field(
        False,
        description=(
            "Whether to cleanup the trigger samples by deleting the directory after the selector has been shut down."
        ),
    )
    ignore_existing_trigger_samples: bool = Field(
        False,
        description=(
            "Whether to ignore existing trigger samples when starting the selector. If set to false, the trigger "
            "sample directory has to be empty upon startup. May lead to unexpected behaviour if set to true and the "
            "trigger sample directory is not empty (e.g. duplicate trigger sample files)"
        ),
    )


# -------------------------------------------------- TRAINING SERVER ------------------------------------------------- #


class TrainingServerConfig(HostnamePortMixin):
    """Configuration for modyn's training server and its grpc service."""

    ftp_port: str = Field(description="The port of the FDP server used by the trainer_server component.")
    offline_dataset_directory: str = Field(
        description=(
            "The directory where the selected samples are stored when downsampling in Sample-then-batch mode is used."
        )
    )


# ------------------------------------------------ METADATA PROCESSOR ------------------------------------------------ #


class MetadataProcessorConfig(HostnamePortMixin):
    """Configuration for modyn's metadata engine."""


# ---------------------------------------------------- TENSORBOARD --------------------------------------------------- #


class TensorboardConfig(BaseModel):
    """Configuration for modyn's Tensorboard."""

    port: str = Field(description="The port on which tensorboard is run.")


# ---------------------------------------------------- SUPERVISOR ---------------------------------------------------- #


class SupervisorConfig(HostnamePortMixin):
    """Configuration for the modyn's supervisor."""

    eval_directory: str | Path = Field(description="The directory to store the evaluation results.")


# ------------------------------------------------------ CONFIG ------------------------------------------------------ #


class ModynConfig(BaseModel):
    """Configuration for the Modyn system.

    Please adapt the fields as needed.
    """

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    storage: StorageConfig

    # model is a reserved keyword in Pydantic, so we use modyn_model instead
    modyn_model_storage: ModelStorageConfig = Field(alias="model_storage")

    evaluator: EvaluatorConfig
    metadata_database: MetadataDatabaseConfig
    metadata_processor: MetadataProcessorConfig | None = Field(None)
    selector: SelectorConfig
    trainer_server: TrainingServerConfig
    tensorboard: TensorboardConfig | None = Field(None)
    supervisor: SupervisorConfig | None = Field(None)
