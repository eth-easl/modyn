"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""

import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing.final
class JsonString(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALUE_FIELD_NUMBER: builtins.int
    value: builtins.str
    def __init__(
        self,
        *,
        value: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["value", b"value"]) -> None: ...

global___JsonString = JsonString

@typing.final
class PythonString(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALUE_FIELD_NUMBER: builtins.int
    value: builtins.str
    def __init__(
        self,
        *,
        value: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["value", b"value"]) -> None: ...

global___PythonString = PythonString

@typing.final
class Data(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATASET_ID_FIELD_NUMBER: builtins.int
    NUM_DATALOADERS_FIELD_NUMBER: builtins.int
    dataset_id: builtins.str
    num_dataloaders: builtins.int
    def __init__(
        self,
        *,
        dataset_id: builtins.str = ...,
        num_dataloaders: builtins.int = ...,
    ) -> None: ...
    def ClearField(
        self, field_name: typing.Literal["dataset_id", b"dataset_id", "num_dataloaders", b"num_dataloaders"]
    ) -> None: ...

global___Data = Data

@typing.final
class TrainerAvailableRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___TrainerAvailableRequest = TrainerAvailableRequest

@typing.final
class TrainerAvailableResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    AVAILABLE_FIELD_NUMBER: builtins.int
    available: builtins.bool
    def __init__(
        self,
        *,
        available: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["available", b"available"]) -> None: ...

global___TrainerAvailableResponse = TrainerAvailableResponse

@typing.final
class CheckpointInfo(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    CHECKPOINT_INTERVAL_FIELD_NUMBER: builtins.int
    CHECKPOINT_PATH_FIELD_NUMBER: builtins.int
    checkpoint_interval: builtins.int
    checkpoint_path: builtins.str
    def __init__(
        self,
        *,
        checkpoint_interval: builtins.int = ...,
        checkpoint_path: builtins.str = ...,
    ) -> None: ...
    def ClearField(
        self,
        field_name: typing.Literal[
            "checkpoint_interval", b"checkpoint_interval", "checkpoint_path", b"checkpoint_path"
        ],
    ) -> None: ...

global___CheckpointInfo = CheckpointInfo

@typing.final
class StartTrainingRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_ID_FIELD_NUMBER: builtins.int
    TRIGGER_ID_FIELD_NUMBER: builtins.int
    DEVICE_FIELD_NUMBER: builtins.int
    USE_PRETRAINED_MODEL_FIELD_NUMBER: builtins.int
    LOAD_OPTIMIZER_STATE_FIELD_NUMBER: builtins.int
    PRETRAINED_MODEL_ID_FIELD_NUMBER: builtins.int
    BATCH_SIZE_FIELD_NUMBER: builtins.int
    TORCH_OPTIMIZERS_CONFIGURATION_FIELD_NUMBER: builtins.int
    TORCH_CRITERION_FIELD_NUMBER: builtins.int
    CRITERION_PARAMETERS_FIELD_NUMBER: builtins.int
    DATA_INFO_FIELD_NUMBER: builtins.int
    CHECKPOINT_INFO_FIELD_NUMBER: builtins.int
    BYTES_PARSER_FIELD_NUMBER: builtins.int
    TRANSFORM_LIST_FIELD_NUMBER: builtins.int
    LR_SCHEDULER_FIELD_NUMBER: builtins.int
    LABEL_TRANSFORMER_FIELD_NUMBER: builtins.int
    GRAD_SCALER_CONFIGURATION_FIELD_NUMBER: builtins.int
    EPOCHS_PER_TRIGGER_FIELD_NUMBER: builtins.int
    NUM_PREFETCHED_PARTITIONS_FIELD_NUMBER: builtins.int
    PARALLEL_PREFETCH_REQUESTS_FIELD_NUMBER: builtins.int
    SEED_FIELD_NUMBER: builtins.int
    TOKENIZER_FIELD_NUMBER: builtins.int
    NUM_SAMPLES_TO_PASS_FIELD_NUMBER: builtins.int
    SHUFFLE_FIELD_NUMBER: builtins.int
    ENABLE_ACCURATE_GPU_MEASUREMENTS_FIELD_NUMBER: builtins.int
    RECORD_LOSS_EVERY_FIELD_NUMBER: builtins.int
    DROP_LAST_BATCH_FIELD_NUMBER: builtins.int
    TOKENIZER_SEQUENCE_LENGTH_FIELD_NUMBER: builtins.int
    pipeline_id: builtins.int
    trigger_id: builtins.int
    device: builtins.str
    use_pretrained_model: builtins.bool
    load_optimizer_state: builtins.bool
    pretrained_model_id: builtins.int
    batch_size: builtins.int
    torch_criterion: builtins.str
    epochs_per_trigger: builtins.int
    num_prefetched_partitions: builtins.int
    parallel_prefetch_requests: builtins.int
    seed: builtins.int
    num_samples_to_pass: builtins.int
    shuffle: builtins.bool
    enable_accurate_gpu_measurements: builtins.bool
    record_loss_every: builtins.int
    drop_last_batch: builtins.bool
    tokenizer_sequence_length: builtins.int
    @property
    def torch_optimizers_configuration(self) -> global___JsonString: ...
    @property
    def criterion_parameters(self) -> global___JsonString: ...
    @property
    def data_info(self) -> global___Data: ...
    @property
    def checkpoint_info(self) -> global___CheckpointInfo: ...
    @property
    def bytes_parser(self) -> global___PythonString: ...
    @property
    def transform_list(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]: ...
    @property
    def lr_scheduler(self) -> global___JsonString: ...
    @property
    def label_transformer(self) -> global___PythonString: ...
    @property
    def grad_scaler_configuration(self) -> global___JsonString: ...
    @property
    def tokenizer(self) -> global___PythonString: ...
    def __init__(
        self,
        *,
        pipeline_id: builtins.int = ...,
        trigger_id: builtins.int = ...,
        device: builtins.str = ...,
        use_pretrained_model: builtins.bool = ...,
        load_optimizer_state: builtins.bool = ...,
        pretrained_model_id: builtins.int = ...,
        batch_size: builtins.int = ...,
        torch_optimizers_configuration: global___JsonString | None = ...,
        torch_criterion: builtins.str = ...,
        criterion_parameters: global___JsonString | None = ...,
        data_info: global___Data | None = ...,
        checkpoint_info: global___CheckpointInfo | None = ...,
        bytes_parser: global___PythonString | None = ...,
        transform_list: collections.abc.Iterable[builtins.str] | None = ...,
        lr_scheduler: global___JsonString | None = ...,
        label_transformer: global___PythonString | None = ...,
        grad_scaler_configuration: global___JsonString | None = ...,
        epochs_per_trigger: builtins.int = ...,
        num_prefetched_partitions: builtins.int = ...,
        parallel_prefetch_requests: builtins.int = ...,
        seed: builtins.int | None = ...,
        tokenizer: global___PythonString | None = ...,
        num_samples_to_pass: builtins.int = ...,
        shuffle: builtins.bool = ...,
        enable_accurate_gpu_measurements: builtins.bool = ...,
        record_loss_every: builtins.int = ...,
        drop_last_batch: builtins.bool = ...,
        tokenizer_sequence_length: builtins.int = ...,
    ) -> None: ...
    def HasField(
        self,
        field_name: typing.Literal[
            "_seed",
            b"_seed",
            "_tokenizer",
            b"_tokenizer",
            "bytes_parser",
            b"bytes_parser",
            "checkpoint_info",
            b"checkpoint_info",
            "criterion_parameters",
            b"criterion_parameters",
            "data_info",
            b"data_info",
            "grad_scaler_configuration",
            b"grad_scaler_configuration",
            "label_transformer",
            b"label_transformer",
            "lr_scheduler",
            b"lr_scheduler",
            "seed",
            b"seed",
            "tokenizer",
            b"tokenizer",
            "torch_optimizers_configuration",
            b"torch_optimizers_configuration",
        ],
    ) -> builtins.bool: ...
    def ClearField(
        self,
        field_name: typing.Literal[
            "_seed",
            b"_seed",
            "_tokenizer",
            b"_tokenizer",
            "batch_size",
            b"batch_size",
            "bytes_parser",
            b"bytes_parser",
            "checkpoint_info",
            b"checkpoint_info",
            "criterion_parameters",
            b"criterion_parameters",
            "data_info",
            b"data_info",
            "device",
            b"device",
            "drop_last_batch",
            b"drop_last_batch",
            "enable_accurate_gpu_measurements",
            b"enable_accurate_gpu_measurements",
            "epochs_per_trigger",
            b"epochs_per_trigger",
            "grad_scaler_configuration",
            b"grad_scaler_configuration",
            "label_transformer",
            b"label_transformer",
            "load_optimizer_state",
            b"load_optimizer_state",
            "lr_scheduler",
            b"lr_scheduler",
            "num_prefetched_partitions",
            b"num_prefetched_partitions",
            "num_samples_to_pass",
            b"num_samples_to_pass",
            "parallel_prefetch_requests",
            b"parallel_prefetch_requests",
            "pipeline_id",
            b"pipeline_id",
            "pretrained_model_id",
            b"pretrained_model_id",
            "record_loss_every",
            b"record_loss_every",
            "seed",
            b"seed",
            "shuffle",
            b"shuffle",
            "tokenizer",
            b"tokenizer",
            "tokenizer_sequence_length",
            b"tokenizer_sequence_length",
            "torch_criterion",
            b"torch_criterion",
            "torch_optimizers_configuration",
            b"torch_optimizers_configuration",
            "transform_list",
            b"transform_list",
            "trigger_id",
            b"trigger_id",
            "use_pretrained_model",
            b"use_pretrained_model",
        ],
    ) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing.Literal["_seed", b"_seed"]) -> typing.Literal["seed"] | None: ...
    @typing.overload
    def WhichOneof(
        self, oneof_group: typing.Literal["_tokenizer", b"_tokenizer"]
    ) -> typing.Literal["tokenizer"] | None: ...

global___StartTrainingRequest = StartTrainingRequest

@typing.final
class StartTrainingResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRAINING_STARTED_FIELD_NUMBER: builtins.int
    TRAINING_ID_FIELD_NUMBER: builtins.int
    training_started: builtins.bool
    training_id: builtins.int
    def __init__(
        self,
        *,
        training_started: builtins.bool = ...,
        training_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(
        self, field_name: typing.Literal["training_id", b"training_id", "training_started", b"training_started"]
    ) -> None: ...

global___StartTrainingResponse = StartTrainingResponse

@typing.final
class TrainingStatusRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRAINING_ID_FIELD_NUMBER: builtins.int
    training_id: builtins.int
    def __init__(
        self,
        *,
        training_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["training_id", b"training_id"]) -> None: ...

global___TrainingStatusRequest = TrainingStatusRequest

@typing.final
class TrainingStatusResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALID_FIELD_NUMBER: builtins.int
    IS_RUNNING_FIELD_NUMBER: builtins.int
    IS_TRAINING_FIELD_NUMBER: builtins.int
    STATE_AVAILABLE_FIELD_NUMBER: builtins.int
    BLOCKED_FIELD_NUMBER: builtins.int
    LOG_FIELD_NUMBER: builtins.int
    EXCEPTION_FIELD_NUMBER: builtins.int
    BATCHES_SEEN_FIELD_NUMBER: builtins.int
    SAMPLES_SEEN_FIELD_NUMBER: builtins.int
    DOWNSAMPLING_BATCHES_SEEN_FIELD_NUMBER: builtins.int
    DOWNSAMPLING_SAMPLES_SEEN_FIELD_NUMBER: builtins.int
    valid: builtins.bool
    is_running: builtins.bool
    is_training: builtins.bool
    state_available: builtins.bool
    blocked: builtins.bool
    exception: builtins.str
    batches_seen: builtins.int
    samples_seen: builtins.int
    downsampling_batches_seen: builtins.int
    downsampling_samples_seen: builtins.int
    @property
    def log(self) -> global___JsonString: ...
    def __init__(
        self,
        *,
        valid: builtins.bool = ...,
        is_running: builtins.bool = ...,
        is_training: builtins.bool = ...,
        state_available: builtins.bool = ...,
        blocked: builtins.bool = ...,
        log: global___JsonString | None = ...,
        exception: builtins.str | None = ...,
        batches_seen: builtins.int | None = ...,
        samples_seen: builtins.int | None = ...,
        downsampling_batches_seen: builtins.int | None = ...,
        downsampling_samples_seen: builtins.int | None = ...,
    ) -> None: ...
    def HasField(
        self,
        field_name: typing.Literal[
            "_batches_seen",
            b"_batches_seen",
            "_downsampling_batches_seen",
            b"_downsampling_batches_seen",
            "_downsampling_samples_seen",
            b"_downsampling_samples_seen",
            "_exception",
            b"_exception",
            "_samples_seen",
            b"_samples_seen",
            "batches_seen",
            b"batches_seen",
            "downsampling_batches_seen",
            b"downsampling_batches_seen",
            "downsampling_samples_seen",
            b"downsampling_samples_seen",
            "exception",
            b"exception",
            "log",
            b"log",
            "samples_seen",
            b"samples_seen",
        ],
    ) -> builtins.bool: ...
    def ClearField(
        self,
        field_name: typing.Literal[
            "_batches_seen",
            b"_batches_seen",
            "_downsampling_batches_seen",
            b"_downsampling_batches_seen",
            "_downsampling_samples_seen",
            b"_downsampling_samples_seen",
            "_exception",
            b"_exception",
            "_samples_seen",
            b"_samples_seen",
            "batches_seen",
            b"batches_seen",
            "blocked",
            b"blocked",
            "downsampling_batches_seen",
            b"downsampling_batches_seen",
            "downsampling_samples_seen",
            b"downsampling_samples_seen",
            "exception",
            b"exception",
            "is_running",
            b"is_running",
            "is_training",
            b"is_training",
            "log",
            b"log",
            "samples_seen",
            b"samples_seen",
            "state_available",
            b"state_available",
            "valid",
            b"valid",
        ],
    ) -> None: ...
    @typing.overload
    def WhichOneof(
        self, oneof_group: typing.Literal["_batches_seen", b"_batches_seen"]
    ) -> typing.Literal["batches_seen"] | None: ...
    @typing.overload
    def WhichOneof(
        self, oneof_group: typing.Literal["_downsampling_batches_seen", b"_downsampling_batches_seen"]
    ) -> typing.Literal["downsampling_batches_seen"] | None: ...
    @typing.overload
    def WhichOneof(
        self, oneof_group: typing.Literal["_downsampling_samples_seen", b"_downsampling_samples_seen"]
    ) -> typing.Literal["downsampling_samples_seen"] | None: ...
    @typing.overload
    def WhichOneof(
        self, oneof_group: typing.Literal["_exception", b"_exception"]
    ) -> typing.Literal["exception"] | None: ...
    @typing.overload
    def WhichOneof(
        self, oneof_group: typing.Literal["_samples_seen", b"_samples_seen"]
    ) -> typing.Literal["samples_seen"] | None: ...

global___TrainingStatusResponse = TrainingStatusResponse

@typing.final
class StoreFinalModelRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRAINING_ID_FIELD_NUMBER: builtins.int
    training_id: builtins.int
    def __init__(
        self,
        *,
        training_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["training_id", b"training_id"]) -> None: ...

global___StoreFinalModelRequest = StoreFinalModelRequest

@typing.final
class StoreFinalModelResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALID_STATE_FIELD_NUMBER: builtins.int
    MODEL_ID_FIELD_NUMBER: builtins.int
    valid_state: builtins.bool
    model_id: builtins.int
    def __init__(
        self,
        *,
        valid_state: builtins.bool = ...,
        model_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(
        self, field_name: typing.Literal["model_id", b"model_id", "valid_state", b"valid_state"]
    ) -> None: ...

global___StoreFinalModelResponse = StoreFinalModelResponse

@typing.final
class GetLatestModelRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRAINING_ID_FIELD_NUMBER: builtins.int
    training_id: builtins.int
    def __init__(
        self,
        *,
        training_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["training_id", b"training_id"]) -> None: ...

global___GetLatestModelRequest = GetLatestModelRequest

@typing.final
class GetLatestModelResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALID_STATE_FIELD_NUMBER: builtins.int
    MODEL_PATH_FIELD_NUMBER: builtins.int
    valid_state: builtins.bool
    model_path: builtins.str
    def __init__(
        self,
        *,
        valid_state: builtins.bool = ...,
        model_path: builtins.str = ...,
    ) -> None: ...
    def ClearField(
        self, field_name: typing.Literal["model_path", b"model_path", "valid_state", b"valid_state"]
    ) -> None: ...

global___GetLatestModelResponse = GetLatestModelResponse
