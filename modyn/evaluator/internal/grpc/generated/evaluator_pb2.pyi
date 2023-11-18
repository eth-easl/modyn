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
class TriggerTrainingSetInfo(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_ID_FIELD_NUMBER: builtins.int
    TRIGGER_ID_FIELD_NUMBER: builtins.int
    NUM_PREFETCHED_PARTITIONS_FIELD_NUMBER: builtins.int
    PARALLEL_PREFETCH_REQUESTS_FIELD_NUMBER: builtins.int
    pipeline_id: builtins.int
    trigger_id: builtins.int
    num_prefetched_partitions: builtins.int
    parallel_prefetch_requests: builtins.int
    def __init__(
        self,
        *,
        pipeline_id: builtins.int = ...,
        trigger_id: builtins.int = ...,
        num_prefetched_partitions: builtins.int = ...,
        parallel_prefetch_requests: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["num_prefetched_partitions", b"num_prefetched_partitions", "parallel_prefetch_requests", b"parallel_prefetch_requests", "pipeline_id", b"pipeline_id", "trigger_id", b"trigger_id"]) -> None: ...

global___TriggerTrainingSetInfo = TriggerTrainingSetInfo

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
class MetricConfiguration(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAME_FIELD_NUMBER: builtins.int
    CONFIG_FIELD_NUMBER: builtins.int
    EVALUATION_TRANSFORMER_FIELD_NUMBER: builtins.int
    name: builtins.str
    @property
    def config(self) -> global___JsonString: ...
    @property
    def evaluation_transformer(self) -> global___PythonString: ...
    def __init__(
        self,
        *,
        name: builtins.str = ...,
        config: global___JsonString | None = ...,
        evaluation_transformer: global___PythonString | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["config", b"config", "evaluation_transformer", b"evaluation_transformer"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["config", b"config", "evaluation_transformer", b"evaluation_transformer", "name", b"name"]) -> None: ...

global___MetricConfiguration = MetricConfiguration

@typing_extensions.final
class EvaluateModelRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    MODEL_ID_FIELD_NUMBER: builtins.int
    DATASET_INFO_FIELD_NUMBER: builtins.int
    DEVICE_FIELD_NUMBER: builtins.int
    BATCH_SIZE_FIELD_NUMBER: builtins.int
    METRICS_FIELD_NUMBER: builtins.int
    TRANSFORM_LIST_FIELD_NUMBER: builtins.int
    BYTES_PARSER_FIELD_NUMBER: builtins.int
    LABEL_TRANSFORMER_FIELD_NUMBER: builtins.int
    TRIGGER_TRAINING_SET_INFO_FIELD_NUMBER: builtins.int
    model_id: builtins.int
    @property
    def dataset_info(self) -> global___DatasetInfo: ...
    device: builtins.str
    batch_size: builtins.int
    @property
    def metrics(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___MetricConfiguration]: ...
    @property
    def transform_list(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]: ...
    @property
    def bytes_parser(self) -> global___PythonString: ...
    @property
    def label_transformer(self) -> global___PythonString: ...
    @property
    def trigger_training_set_info(self) -> global___TriggerTrainingSetInfo: ...
    def __init__(
        self,
        *,
        model_id: builtins.int = ...,
        dataset_info: global___DatasetInfo | None = ...,
        device: builtins.str = ...,
        batch_size: builtins.int = ...,
        metrics: collections.abc.Iterable[global___MetricConfiguration] | None = ...,
        transform_list: collections.abc.Iterable[builtins.str] | None = ...,
        bytes_parser: global___PythonString | None = ...,
        label_transformer: global___PythonString | None = ...,
        trigger_training_set_info: global___TriggerTrainingSetInfo | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_trigger_training_set_info", b"_trigger_training_set_info", "bytes_parser", b"bytes_parser", "dataset_info", b"dataset_info", "label_transformer", b"label_transformer", "trigger_training_set_info", b"trigger_training_set_info"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_trigger_training_set_info", b"_trigger_training_set_info", "batch_size", b"batch_size", "bytes_parser", b"bytes_parser", "dataset_info", b"dataset_info", "device", b"device", "label_transformer", b"label_transformer", "metrics", b"metrics", "model_id", b"model_id", "transform_list", b"transform_list", "trigger_training_set_info", b"trigger_training_set_info"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_trigger_training_set_info", b"_trigger_training_set_info"]) -> typing_extensions.Literal["trigger_training_set_info"] | None: ...

global___EvaluateModelRequest = EvaluateModelRequest

@typing_extensions.final
class EvaluateModelResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    EVALUATION_STARTED_FIELD_NUMBER: builtins.int
    EVALUATION_ID_FIELD_NUMBER: builtins.int
    DATASET_SIZE_FIELD_NUMBER: builtins.int
    evaluation_started: builtins.bool
    evaluation_id: builtins.int
    dataset_size: builtins.int
    def __init__(
        self,
        *,
        evaluation_started: builtins.bool = ...,
        evaluation_id: builtins.int = ...,
        dataset_size: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["dataset_size", b"dataset_size", "evaluation_id", b"evaluation_id", "evaluation_started", b"evaluation_started"]) -> None: ...

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
class EvaluationResultRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    EVALUATION_ID_FIELD_NUMBER: builtins.int
    evaluation_id: builtins.int
    def __init__(
        self,
        *,
        evaluation_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["evaluation_id", b"evaluation_id"]) -> None: ...

global___EvaluationResultRequest = EvaluationResultRequest

@typing_extensions.final
class EvaluationResultResponse(google.protobuf.message.Message):
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

global___EvaluationResultResponse = EvaluationResultResponse
