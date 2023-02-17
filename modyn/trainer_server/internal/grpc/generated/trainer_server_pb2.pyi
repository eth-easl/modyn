"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import sys
import typing

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class JsonString(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALUE_FIELD_NUMBER: builtins.int
    value: builtins.str
    def __init__(
        self,
        *,
        value: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["value", b"value"]) -> None: ...

global___JsonString = JsonString

@typing_extensions.final
class PythonString(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALUE_FIELD_NUMBER: builtins.int
    value: builtins.str
    def __init__(
        self,
        *,
        value: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["value", b"value"]) -> None: ...

global___PythonString = PythonString

@typing_extensions.final
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
    def ClearField(self, field_name: typing_extensions.Literal["dataset_id", b"dataset_id", "num_dataloaders", b"num_dataloaders"]) -> None: ...

global___Data = Data

@typing_extensions.final
class TrainerAvailableRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___TrainerAvailableRequest = TrainerAvailableRequest

@typing_extensions.final
class TrainerAvailableResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    AVAILABLE_FIELD_NUMBER: builtins.int
    available: builtins.bool
    def __init__(
        self,
        *,
        available: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["available", b"available"]) -> None: ...

global___TrainerAvailableResponse = TrainerAvailableResponse

@typing_extensions.final
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
    def ClearField(self, field_name: typing_extensions.Literal["checkpoint_interval", b"checkpoint_interval", "checkpoint_path", b"checkpoint_path"]) -> None: ...

global___CheckpointInfo = CheckpointInfo

@typing_extensions.final
class StartTrainingRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_ID_FIELD_NUMBER: builtins.int
    TRIGGER_ID_FIELD_NUMBER: builtins.int
    DEVICE_FIELD_NUMBER: builtins.int
    MODEL_ID_FIELD_NUMBER: builtins.int
    MODEL_CONFIGURATION_FIELD_NUMBER: builtins.int
    USE_PRETRAINED_MODEL_FIELD_NUMBER: builtins.int
    LOAD_OPTIMIZER_STATE_FIELD_NUMBER: builtins.int
    PRETRAINED_MODEL_FIELD_NUMBER: builtins.int
    BATCH_SIZE_FIELD_NUMBER: builtins.int
    TORCH_OPTIMIZERS_FIELD_NUMBER: builtins.int
    OPTIMIZER_PARAMETERS_FIELD_NUMBER: builtins.int
    TORCH_CRITERION_FIELD_NUMBER: builtins.int
    CRITERION_PARAMETERS_FIELD_NUMBER: builtins.int
    DATA_INFO_FIELD_NUMBER: builtins.int
    CHECKPOINT_INFO_FIELD_NUMBER: builtins.int
    BYTES_PARSER_FIELD_NUMBER: builtins.int
    TRANSFORM_LIST_FIELD_NUMBER: builtins.int
    TORCH_LR_SCHEDULER_FIELD_NUMBER: builtins.int
    CUSTOM_LR_SCHEDULER_FIELD_NUMBER: builtins.int
    LR_SCHEDULER_CONFIGS_FIELD_NUMBER: builtins.int
    pipeline_id: builtins.int
    trigger_id: builtins.int
    device: builtins.str
    model_id: builtins.str
    @property
    def model_configuration(self) -> global___JsonString: ...
    use_pretrained_model: builtins.bool
    load_optimizer_state: builtins.bool
    pretrained_model: builtins.bytes
    batch_size: builtins.int
    @property
    def torch_optimizers(self) -> global___JsonString: ...
    @property
    def optimizer_parameters(self) -> global___JsonString: ...
    torch_criterion: builtins.str
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
    torch_lr_scheduler: builtins.str
    custom_lr_scheduler: builtins.str
    @property
    def lr_scheduler_configs(self) -> global___JsonString: ...
    def __init__(
        self,
        *,
        pipeline_id: builtins.int = ...,
        trigger_id: builtins.int = ...,
        device: builtins.str = ...,
        model_id: builtins.str = ...,
        model_configuration: global___JsonString | None = ...,
        use_pretrained_model: builtins.bool = ...,
        load_optimizer_state: builtins.bool = ...,
        pretrained_model: builtins.bytes = ...,
        batch_size: builtins.int = ...,
        torch_optimizers: global___JsonString | None = ...,
        optimizer_parameters: global___JsonString | None = ...,
        torch_criterion: builtins.str = ...,
        criterion_parameters: global___JsonString | None = ...,
        data_info: global___Data | None = ...,
        checkpoint_info: global___CheckpointInfo | None = ...,
        bytes_parser: global___PythonString | None = ...,
        transform_list: collections.abc.Iterable[builtins.str] | None = ...,
        torch_lr_scheduler: builtins.str = ...,
        custom_lr_scheduler: builtins.str = ...,
        lr_scheduler_configs: global___JsonString | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["bytes_parser", b"bytes_parser", "checkpoint_info", b"checkpoint_info", "criterion_parameters", b"criterion_parameters", "data_info", b"data_info", "lr_scheduler_configs", b"lr_scheduler_configs", "model_configuration", b"model_configuration", "optimizer_parameters", b"optimizer_parameters", "torch_optimizers", b"torch_optimizers"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["batch_size", b"batch_size", "bytes_parser", b"bytes_parser", "checkpoint_info", b"checkpoint_info", "criterion_parameters", b"criterion_parameters", "custom_lr_scheduler", b"custom_lr_scheduler", "data_info", b"data_info", "device", b"device", "load_optimizer_state", b"load_optimizer_state", "lr_scheduler_configs", b"lr_scheduler_configs", "model_configuration", b"model_configuration", "model_id", b"model_id", "optimizer_parameters", b"optimizer_parameters", "pipeline_id", b"pipeline_id", "pretrained_model", b"pretrained_model", "torch_criterion", b"torch_criterion", "torch_lr_scheduler", b"torch_lr_scheduler", "torch_optimizers", b"torch_optimizers", "transform_list", b"transform_list", "trigger_id", b"trigger_id", "use_pretrained_model", b"use_pretrained_model"]) -> None: ...

global___StartTrainingRequest = StartTrainingRequest

@typing_extensions.final
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
    def ClearField(self, field_name: typing_extensions.Literal["training_id", b"training_id", "training_started", b"training_started"]) -> None: ...

global___StartTrainingResponse = StartTrainingResponse

@typing_extensions.final
class TrainingStatusRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRAINING_ID_FIELD_NUMBER: builtins.int
    training_id: builtins.int
    def __init__(
        self,
        *,
        training_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["training_id", b"training_id"]) -> None: ...

global___TrainingStatusRequest = TrainingStatusRequest

@typing_extensions.final
class TrainingStatusResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALID_FIELD_NUMBER: builtins.int
    IS_RUNNING_FIELD_NUMBER: builtins.int
    STATE_AVAILABLE_FIELD_NUMBER: builtins.int
    BLOCKED_FIELD_NUMBER: builtins.int
    EXCEPTION_FIELD_NUMBER: builtins.int
    BATCHES_SEEN_FIELD_NUMBER: builtins.int
    SAMPLES_SEEN_FIELD_NUMBER: builtins.int
    valid: builtins.bool
    is_running: builtins.bool
    state_available: builtins.bool
    blocked: builtins.bool
    exception: builtins.str
    batches_seen: builtins.int
    samples_seen: builtins.int
    def __init__(
        self,
        *,
        valid: builtins.bool = ...,
        is_running: builtins.bool = ...,
        state_available: builtins.bool = ...,
        blocked: builtins.bool = ...,
        exception: builtins.str | None = ...,
        batches_seen: builtins.int | None = ...,
        samples_seen: builtins.int | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_batches_seen", b"_batches_seen", "_exception", b"_exception", "_samples_seen", b"_samples_seen", "batches_seen", b"batches_seen", "exception", b"exception", "samples_seen", b"samples_seen"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_batches_seen", b"_batches_seen", "_exception", b"_exception", "_samples_seen", b"_samples_seen", "batches_seen", b"batches_seen", "blocked", b"blocked", "exception", b"exception", "is_running", b"is_running", "samples_seen", b"samples_seen", "state_available", b"state_available", "valid", b"valid"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_batches_seen", b"_batches_seen"]) -> typing_extensions.Literal["batches_seen"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_exception", b"_exception"]) -> typing_extensions.Literal["exception"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_samples_seen", b"_samples_seen"]) -> typing_extensions.Literal["samples_seen"] | None: ...

global___TrainingStatusResponse = TrainingStatusResponse

@typing_extensions.final
class GetFinalModelRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRAINING_ID_FIELD_NUMBER: builtins.int
    training_id: builtins.int
    def __init__(
        self,
        *,
        training_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["training_id", b"training_id"]) -> None: ...

global___GetFinalModelRequest = GetFinalModelRequest

@typing_extensions.final
class GetFinalModelResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALID_STATE_FIELD_NUMBER: builtins.int
    STATE_FIELD_NUMBER: builtins.int
    valid_state: builtins.bool
    state: builtins.bytes
    def __init__(
        self,
        *,
        valid_state: builtins.bool = ...,
        state: builtins.bytes = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["state", b"state", "valid_state", b"valid_state"]) -> None: ...

global___GetFinalModelResponse = GetFinalModelResponse

@typing_extensions.final
class GetLatestModelRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRAINING_ID_FIELD_NUMBER: builtins.int
    training_id: builtins.int
    def __init__(
        self,
        *,
        training_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["training_id", b"training_id"]) -> None: ...

global___GetLatestModelRequest = GetLatestModelRequest

@typing_extensions.final
class GetLatestModelResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALID_STATE_FIELD_NUMBER: builtins.int
    STATE_FIELD_NUMBER: builtins.int
    valid_state: builtins.bool
    state: builtins.bytes
    def __init__(
        self,
        *,
        valid_state: builtins.bool = ...,
        state: builtins.bytes = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["state", b"state", "valid_state", b"valid_state"]) -> None: ...

global___GetLatestModelResponse = GetLatestModelResponse
