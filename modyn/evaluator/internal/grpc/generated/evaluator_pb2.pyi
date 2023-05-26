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

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class DatasetInfo(google.protobuf.message.Message):
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

global___DatasetInfo = DatasetInfo

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
class EvaluateModelRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRAINED_MODEL_ID_FIELD_NUMBER: builtins.int
    DATASET_INFO_FIELD_NUMBER: builtins.int
    DEVICE_FIELD_NUMBER: builtins.int
    AMP_FIELD_NUMBER: builtins.int
    BATCH_SIZE_FIELD_NUMBER: builtins.int
    EVALUATION_LAYER_FIELD_NUMBER: builtins.int
    METRICS_FIELD_NUMBER: builtins.int
    MODEL_ID_FIELD_NUMBER: builtins.int
    MODEL_CONFIGURATION_FIELD_NUMBER: builtins.int
    TRANSFORM_LIST_FIELD_NUMBER: builtins.int
    BYTES_PARSER_FIELD_NUMBER: builtins.int
    LABEL_TRANSFORMER_FIELD_NUMBER: builtins.int
    trained_model_id: builtins.int
    @property
    def dataset_info(self) -> global___DatasetInfo: ...
    device: builtins.str
    amp: builtins.bool
    batch_size: builtins.int
    @property
    def evaluation_layer(self) -> global___PythonString: ...
    @property
    def metrics(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]: ...
    model_id: builtins.str
    @property
    def model_configuration(self) -> global___JsonString: ...
    @property
    def transform_list(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]: ...
    @property
    def bytes_parser(self) -> global___PythonString: ...
    @property
    def label_transformer(self) -> global___PythonString: ...
    def __init__(
        self,
        *,
        trained_model_id: builtins.int = ...,
        dataset_info: global___DatasetInfo | None = ...,
        device: builtins.str = ...,
        amp: builtins.bool = ...,
        batch_size: builtins.int = ...,
        evaluation_layer: global___PythonString | None = ...,
        metrics: collections.abc.Iterable[builtins.str] | None = ...,
        model_id: builtins.str = ...,
        model_configuration: global___JsonString | None = ...,
        transform_list: collections.abc.Iterable[builtins.str] | None = ...,
        bytes_parser: global___PythonString | None = ...,
        label_transformer: global___PythonString | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["bytes_parser", b"bytes_parser", "dataset_info", b"dataset_info", "evaluation_layer", b"evaluation_layer", "label_transformer", b"label_transformer", "model_configuration", b"model_configuration"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["amp", b"amp", "batch_size", b"batch_size", "bytes_parser", b"bytes_parser", "dataset_info", b"dataset_info", "device", b"device", "evaluation_layer", b"evaluation_layer", "label_transformer", b"label_transformer", "metrics", b"metrics", "model_configuration", b"model_configuration", "model_id", b"model_id", "trained_model_id", b"trained_model_id", "transform_list", b"transform_list"]) -> None: ...

global___EvaluateModelRequest = EvaluateModelRequest

@typing_extensions.final
class EvaluateModelResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    EVALUATION_STARTED_FIELD_NUMBER: builtins.int
    EVALUATION_ID_FIELD_NUMBER: builtins.int
    evaluation_started: builtins.bool
    evaluation_id: builtins.int
    def __init__(
        self,
        *,
        evaluation_started: builtins.bool = ...,
        evaluation_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["evaluation_id", b"evaluation_id", "evaluation_started", b"evaluation_started"]) -> None: ...

global___EvaluateModelResponse = EvaluateModelResponse

@typing_extensions.final
class EvaluationStatusRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    EVALUATION_ID_FIELD_NUMBER: builtins.int
    evaluation_id: builtins.int
    def __init__(
        self,
        *,
        evaluation_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["evaluation_id", b"evaluation_id"]) -> None: ...

global___EvaluationStatusRequest = EvaluationStatusRequest

@typing_extensions.final
class EvaluationStatusResponse(google.protobuf.message.Message):
    """TODO(#15) check what we need here"""

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
        exception: builtins.str = ...,
        batches_seen: builtins.int = ...,
        samples_seen: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["batches_seen", b"batches_seen", "blocked", b"blocked", "exception", b"exception", "is_running", b"is_running", "samples_seen", b"samples_seen", "state_available", b"state_available", "valid", b"valid"]) -> None: ...

global___EvaluationStatusResponse = EvaluationStatusResponse

@typing_extensions.final
class EvaluationData(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    METRIC_FIELD_NUMBER: builtins.int
    RESULT_FIELD_NUMBER: builtins.int
    metric: builtins.str
    result: builtins.float
    def __init__(
        self,
        *,
        metric: builtins.str = ...,
        result: builtins.float = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["metric", b"metric", "result", b"result"]) -> None: ...

global___EvaluationData = EvaluationData

@typing_extensions.final
class FinalEvaluationRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    EVALUATION_ID_FIELD_NUMBER: builtins.int
    evaluation_id: builtins.int
    def __init__(
        self,
        *,
        evaluation_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["evaluation_id", b"evaluation_id"]) -> None: ...

global___FinalEvaluationRequest = FinalEvaluationRequest

@typing_extensions.final
class FinalEvaluationResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALID_FIELD_NUMBER: builtins.int
    EVALUATION_DATA_FIELD_NUMBER: builtins.int
    valid: builtins.bool
    @property
    def evaluation_data(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___EvaluationData]: ...
    def __init__(
        self,
        *,
        valid: builtins.bool = ...,
        evaluation_data: collections.abc.Iterable[global___EvaluationData] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["evaluation_data", b"evaluation_data", "valid", b"valid"]) -> None: ...

global___FinalEvaluationResponse = FinalEvaluationResponse
