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
class RegisterTrainServerRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRAINING_ID_FIELD_NUMBER: builtins.int
    MODEL_ID_FIELD_NUMBER: builtins.int
    TORCH_OPTIMIZER_FIELD_NUMBER: builtins.int
    BATCH_SIZE_FIELD_NUMBER: builtins.int
    TORCH_CRITERION_FIELD_NUMBER: builtins.int
    CRITERION_PARAMETERS_FIELD_NUMBER: builtins.int
    OPTIMIZER_PARAMETERS_FIELD_NUMBER: builtins.int
    MODEL_CONFIGURATION_FIELD_NUMBER: builtins.int
    DATA_INFO_FIELD_NUMBER: builtins.int
    CHECKPOINT_INFO_FIELD_NUMBER: builtins.int
    TRANSFORM_LIST_FIELD_NUMBER: builtins.int
    training_id: builtins.int
    model_id: builtins.str
    torch_optimizer: builtins.str
    batch_size: builtins.int
    torch_criterion: builtins.str
    @property
    def criterion_parameters(self) -> global___JsonString: ...
    @property
    def optimizer_parameters(self) -> global___JsonString: ...
    @property
    def model_configuration(self) -> global___JsonString: ...
    @property
    def data_info(self) -> global___Data: ...
    @property
    def checkpoint_info(self) -> global___CheckpointInfo: ...
    @property
    def transform_list(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]: ...
    def __init__(
        self,
        *,
        training_id: builtins.int = ...,
        model_id: builtins.str = ...,
        torch_optimizer: builtins.str = ...,
        batch_size: builtins.int = ...,
        torch_criterion: builtins.str = ...,
        criterion_parameters: global___JsonString | None = ...,
        optimizer_parameters: global___JsonString | None = ...,
        model_configuration: global___JsonString | None = ...,
        data_info: global___Data | None = ...,
        checkpoint_info: global___CheckpointInfo | None = ...,
        transform_list: collections.abc.Iterable[builtins.str] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["checkpoint_info", b"checkpoint_info", "criterion_parameters", b"criterion_parameters", "data_info", b"data_info", "model_configuration", b"model_configuration", "optimizer_parameters", b"optimizer_parameters"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["batch_size", b"batch_size", "checkpoint_info", b"checkpoint_info", "criterion_parameters", b"criterion_parameters", "data_info", b"data_info", "model_configuration", b"model_configuration", "model_id", b"model_id", "optimizer_parameters", b"optimizer_parameters", "torch_criterion", b"torch_criterion", "torch_optimizer", b"torch_optimizer", "training_id", b"training_id", "transform_list", b"transform_list"]) -> None: ...

global___RegisterTrainServerRequest = RegisterTrainServerRequest

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
class RegisterTrainServerResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SUCCESS_FIELD_NUMBER: builtins.int
    success: builtins.bool
    def __init__(
        self,
        *,
        success: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["success", b"success"]) -> None: ...

global___RegisterTrainServerResponse = RegisterTrainServerResponse

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

    TRAINING_ID_FIELD_NUMBER: builtins.int
    DEVICE_FIELD_NUMBER: builtins.int
    TRAIN_UNTIL_SAMPLE_ID_FIELD_NUMBER: builtins.int
    LOAD_CHECKPOINT_PATH_FIELD_NUMBER: builtins.int
    training_id: builtins.int
    device: builtins.str
    train_until_sample_id: builtins.str
    load_checkpoint_path: builtins.str
    def __init__(
        self,
        *,
        training_id: builtins.int = ...,
        device: builtins.str = ...,
        train_until_sample_id: builtins.str = ...,
        load_checkpoint_path: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["device", b"device", "load_checkpoint_path", b"load_checkpoint_path", "train_until_sample_id", b"train_until_sample_id", "training_id", b"training_id"]) -> None: ...

global___StartTrainingRequest = StartTrainingRequest

@typing_extensions.final
class StartTrainingResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRAINING_STARTED_FIELD_NUMBER: builtins.int
    training_started: builtins.bool
    def __init__(
        self,
        *,
        training_started: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["training_started", b"training_started"]) -> None: ...

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
    STATE_FIELD_NUMBER: builtins.int
    valid: builtins.bool
    is_running: builtins.bool
    state_available: builtins.bool
    blocked: builtins.bool
    exception: builtins.str
    batches_seen: builtins.int
    samples_seen: builtins.int
    state: builtins.bytes
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
        state: builtins.bytes | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_batches_seen", b"_batches_seen", "_exception", b"_exception", "_samples_seen", b"_samples_seen", "_state", b"_state", "batches_seen", b"batches_seen", "exception", b"exception", "samples_seen", b"samples_seen", "state", b"state"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_batches_seen", b"_batches_seen", "_exception", b"_exception", "_samples_seen", b"_samples_seen", "_state", b"_state", "batches_seen", b"batches_seen", "blocked", b"blocked", "exception", b"exception", "is_running", b"is_running", "samples_seen", b"samples_seen", "state", b"state", "state_available", b"state_available", "valid", b"valid"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_batches_seen", b"_batches_seen"]) -> typing_extensions.Literal["batches_seen"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_exception", b"_exception"]) -> typing_extensions.Literal["exception"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_samples_seen", b"_samples_seen"]) -> typing_extensions.Literal["samples_seen"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_state", b"_state"]) -> typing_extensions.Literal["state"] | None: ...

global___TrainingStatusResponse = TrainingStatusResponse
