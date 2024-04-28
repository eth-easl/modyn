from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

# ------------------------------------------------------ MIXINS ------------------------------------------------------ #


class HostnamePortMixin(BaseModel):
    """
    Mixin for hostname and port configuration.
    """

    hostname: str = Field(description="The hostname where the service can be reached.")
    port: str = Field(description="The port where the service can be reached.")


class DatabaseMixin(HostnamePortMixin):
    """
    Mixin for a database configuration.
    """

    drivername: str = Field(description="The drivername to use for the database.")
    username: str = Field(description="The username to use for the database.")
    password: str = Field(description="The password to use for the database.")
    database: str = Field(description="The database to use for the database.")


# ------------------------------------------------------ PROJECT ----------------------------------------------------- #


class ProjectConfig(BaseModel):
    """
    Represents the configuration of a project.
    """

    name: str = Field(description="The name of the project.")
    description: str = Field(description="The description of the project.")
    version: str = Field(description="The version of the project.")


# ------------------------------------------------------ STORAGE ----------------------------------------------------- #

BinaryFileByteOrder = Literal["big", "little"]


class DatasetFileWrapperConfig(BaseModel):
    """
    Represents a dataset file used by modyn.
    """

    file_extension: str = Field(
        description="The file extension of the dataset.", pattern=r"^\..*$"
    )
    label_file_extension: str = Field(
        description="The label file extension of the dataset", pattern=r"$\..*$"
    )

    # [BinaryFileWrapper]
    # TODO: required only conditionally
    record_size: int = Field(
        description="The size of each full record in bytes (label + features)."
    )
    label_size: int = Field(
        description="The size of the label field in bytes for a binary file wrapper."
    )
    byteorder: BinaryFileByteOrder = Field(
        description="The byteorder when reading an integer from multibyte data in a binary file."
    )

    # [CsvFileWrapper]
    separator: str = Field(",", description="The separator used in CSV files.")
    label_index: int = Field(
        description=(
            "Column index of the label. For columns 'width, 'height, 'age', 'label' you should set label_index to 3."
        )
    )
    ignore_first_line: bool = Field(
        False,
        description="If the first line is the table header, you can skip it setting this parameter to True.",
    )
    encoding: str = Field("utf-8", description="Encoding of the CSV files.")
    validate_file_content: bool | None = Field(
        True,
        description=(
            "Whether to validate the file content before inserting the data. It checks that it is a csv, that all "
            "rows are the same size and that the 'label' column exists."
        ),
    )


class DatasetsConfig(BaseModel):
    """
    Configures a dataset to be used by modyn.
    """

    name: str = Field(description="The name of the dataset.")
    description: str = Field(description="The description of the dataset.")
    version: str = Field(description="The version of the dataset.")
    base_path: str = Field(description="The base path of the dataset.")
    filesystem_wrapper_type: str = Field(
        description="The filesystem wrapper type of the dataset."
    )
    file_wrapper_type: str = Field(description="The file wrapper type of the dataset.")
    file_wrapper_config: DatasetFileWrapperConfig = Field(
        description="The file wrapper config of the dataset."
    )
    ignore_last_timestamp: bool | None = Field(
        description=(
            "Whether to ignore the last timestamp when scanning for new files, i.e., if this is set to false, in "
            "case a new file gets added to the storage that has a smaller timestamp than the latest file that the "
            "storage has already processed, the file is not processed."
        )
    )
    file_watcher_interval: int | None = Field(
        description="The interval in seconds in which the file watcher checks for new files."
    )
    selector_batch_size: int = Field(
        True,
        description="The number of samples per which we check for triggers and inform the selector.",
    )


class DatabaseConfig(DatabaseMixin):
    """
    Configuration for modyn's main database.
    """

    hash_partition_modulus: int | None = Field(
        None, description="The modulus to use for the hash partitioning of the samples."
    )


class StorageConfig(BaseModel):
    """
    Configuration for modyn's storage engine.
    """

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
        description=(
            "The number of threads used to get samples from the storage DB. If set to <= 1, multithreaded gets "
            "are disabled."
        )
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
        False,
        description="Enforces fallback insert functionality instead of potentially optimized techniques.",
    )
    file_watcher_watchdog_sleep_time_s: int = Field(
        3,
        description=(
            "The time in seconds the file watcher watchdog sleeps between checking if the file watchers are "
            "still alive."
        ),
    )
    datasets: List[DatasetsConfig] = Field(
        default_factory=list, description="The datasets to use for the storage engine."
    )
    database: DatabaseConfig = Field(description="The database configuration.")


# --------------------------------------------------- MODEL STORAGE -------------------------------------------------- #


class ModelStorageConfig(HostnamePortMixin):
    """
    Configuration for modyn's model storage component and its grpc service.
    """

    ftp_port: str = Field(
        description="The port of the FDP server used by the model_storage component."
    )
    models_directory: str | None = Field(
        description="The directory where we store the trained models."
    )


# ----------------------------------------------------- EVALUATOR ---------------------------------------------------- #


class EvaluatorConfig(HostnamePortMixin):
    """
    Configuration for modyn's evaluation component.
    """


# ---------------------------------------------------- METADATA DB --------------------------------------------------- #


class MetadataDatabaseConfig(DatabaseConfig):
    """
    Configuration for modyn's metadata database.
    """

    hash_partition_modulus: int | None = Field(
        description="The modulus to use for the hash partitioning of the metadata."
    )
    seed: float | None = Field(
        None,
        description="If provided, this number is used to seed the database.",
        ge=-1,
        le=1,
    )


# ----------------------------------------------------- SELECTOR ----------------------------------------------------- #


class SelectorConfig(HostnamePortMixin):
    """
    Configuration for modyn's selector.
    """

    keys_in_selector_cache: int = Field(
        description="How many keys each selector is allowed to cache in memory."
    )
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
    """
    Configuration for modyn's training server and its grpc service.
    """

    ftp_port: str = Field(
        description="The port of the FDP server used by the trainer_server component."
    )
    offline_dataset_directory: str = Field(
        description=(
            "The directory where the selected samples are stored when downsampling in Sample-then-batch mode is used."
        )
    )


# ------------------------------------------------ METADATA PROCESSOR ------------------------------------------------ #


class MetadataProcessorConfig(HostnamePortMixin):
    """
    Configuration for modyn's metadata engine.
    """


# ---------------------------------------------------- TENSORBOARD --------------------------------------------------- #


class TensorboardConfig(BaseModel):
    """
    Configuration for modyn's Tensorboard.
    """

    port: str = Field(description="The port on which tensorboard is run.")


# ---------------------------------------------------- SUPERVISOR ---------------------------------------------------- #


class SupervisorConfig(HostnamePortMixin):
    """
    Configuration for the modyn's supervisor.
    """

    eval_directory: str = Field(
        description="The directory to store the evaluation results."
    )


# ------------------------------------------------------ CONFIG ------------------------------------------------------ #


class ModynConfig(BaseModel):
    """Configuration for the Modyn system. Please adapt the fields as needed."""

    project: ProjectConfig
    storage: StorageConfig

    # model is a reserved keyword in Pydantic, so we use modyn_model instead
    modyn_model_storage: ModelStorageConfig = Field(alias="model_storage")

    evaluator: EvaluatorConfig
    metadata_database: MetadataDatabaseConfig
    metadata_processor: MetadataProcessorConfig | None
    selector: SelectorConfig
    trainer_server: TrainingServerConfig
    tensorboard: TensorboardConfig | None
    supervisor: SupervisorConfig | None
