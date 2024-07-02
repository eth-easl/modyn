"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""

import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import sys
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _EvaluationAbortedReason:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _EvaluationAbortedReasonEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_EvaluationAbortedReason.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    UNKNOWN: _EvaluationAbortedReason.ValueType  # 0
    MODEL_NOT_EXIST_IN_METADATA: _EvaluationAbortedReason.ValueType  # 1
    MODEL_IMPORT_FAILURE: _EvaluationAbortedReason.ValueType  # 2
    MODEL_NOT_EXIST_IN_STORAGE: _EvaluationAbortedReason.ValueType  # 3
    DATASET_NOT_FOUND: _EvaluationAbortedReason.ValueType  # 4
    EMPTY_DATASET: _EvaluationAbortedReason.ValueType  # 5
    DOWNLOAD_MODEL_FAILURE: _EvaluationAbortedReason.ValueType  # 6

class EvaluationAbortedReason(_EvaluationAbortedReason, metaclass=_EvaluationAbortedReasonEnumTypeWrapper): ...

UNKNOWN: EvaluationAbortedReason.ValueType  # 0
MODEL_NOT_EXIST_IN_METADATA: EvaluationAbortedReason.ValueType  # 1
MODEL_IMPORT_FAILURE: EvaluationAbortedReason.ValueType  # 2
MODEL_NOT_EXIST_IN_STORAGE: EvaluationAbortedReason.ValueType  # 3
DATASET_NOT_FOUND: EvaluationAbortedReason.ValueType  # 4
EMPTY_DATASET: EvaluationAbortedReason.ValueType  # 5
DOWNLOAD_MODEL_FAILURE: EvaluationAbortedReason.ValueType  # 6
global___EvaluationAbortedReason = EvaluationAbortedReason

@typing.final
class EvaluationInterval(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    START_TIMESTAMP_FIELD_NUMBER: builtins.int
    END_TIMESTAMP_FIELD_NUMBER: builtins.int
    start_timestamp: builtins.int
    end_timestamp: builtins.int
    def __init__(
        self,
        *,
        start_timestamp: builtins.int | None = ...,
        end_timestamp: builtins.int | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["_end_timestamp", b"_end_timestamp", "_start_timestamp", b"_start_timestamp", "end_timestamp", b"end_timestamp", "start_timestamp", b"start_timestamp"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["_end_timestamp", b"_end_timestamp", "_start_timestamp", b"_start_timestamp", "end_timestamp", b"end_timestamp", "start_timestamp", b"start_timestamp"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing.Literal["_end_timestamp", b"_end_timestamp"]) -> typing.Literal["end_timestamp"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing.Literal["_start_timestamp", b"_start_timestamp"]) -> typing.Literal["start_timestamp"] | None: ...

global___EvaluationInterval = EvaluationInterval

@typing.final
class DatasetInfo(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATASET_ID_FIELD_NUMBER: builtins.int
    NUM_DATALOADERS_FIELD_NUMBER: builtins.int
    EVALUATION_INTERVALS_FIELD_NUMBER: builtins.int
    dataset_id: builtins.str
    num_dataloaders: builtins.int
    @property
    def evaluation_intervals(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___EvaluationInterval]: ...
    def __init__(
        self,
        *,
        dataset_id: builtins.str = ...,
        num_dataloaders: builtins.int = ...,
        evaluation_intervals: collections.abc.Iterable[global___EvaluationInterval] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["dataset_id", b"dataset_id", "evaluation_intervals", b"evaluation_intervals", "num_dataloaders", b"num_dataloaders"]) -> None: ...

global___DatasetInfo = DatasetInfo

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
    TOKENIZER_FIELD_NUMBER: builtins.int
    model_id: builtins.int
    device: builtins.str
    batch_size: builtins.int
    @property
    def dataset_info(self) -> global___DatasetInfo: ...
    @property
    def metrics(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___JsonString]: ...
    @property
    def transform_list(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]: ...
    @property
    def bytes_parser(self) -> global___PythonString: ...
    @property
    def label_transformer(self) -> global___PythonString: ...
    @property
    def tokenizer(self) -> global___PythonString: ...
    def __init__(
        self,
        *,
        model_id: builtins.int = ...,
        dataset_info: global___DatasetInfo | None = ...,
        device: builtins.str = ...,
        batch_size: builtins.int = ...,
        metrics: collections.abc.Iterable[global___JsonString] | None = ...,
        transform_list: collections.abc.Iterable[builtins.str] | None = ...,
        bytes_parser: global___PythonString | None = ...,
        label_transformer: global___PythonString | None = ...,
        tokenizer: global___PythonString | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["_tokenizer", b"_tokenizer", "bytes_parser", b"bytes_parser", "dataset_info", b"dataset_info", "label_transformer", b"label_transformer", "tokenizer", b"tokenizer"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["_tokenizer", b"_tokenizer", "batch_size", b"batch_size", "bytes_parser", b"bytes_parser", "dataset_info", b"dataset_info", "device", b"device", "label_transformer", b"label_transformer", "metrics", b"metrics", "model_id", b"model_id", "tokenizer", b"tokenizer", "transform_list", b"transform_list"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["_tokenizer", b"_tokenizer"]) -> typing.Literal["tokenizer"] | None: ...

global___EvaluateModelRequest = EvaluateModelRequest

@typing.final
class EvaluateModelResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    EVALUATION_STARTED_FIELD_NUMBER: builtins.int
    EVALUATION_ID_FIELD_NUMBER: builtins.int
    DATASET_SIZES_FIELD_NUMBER: builtins.int
    EVAL_ABORTED_REASONS_FIELD_NUMBER: builtins.int
    evaluation_started: builtins.bool
    """only when all interval evaluations failed, this field will be set to false"""
    evaluation_id: builtins.int
    @property
    def dataset_sizes(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """non-empty only when evaluation_started is true; in this case it has the same size as the number of intervals"""

    @property
    def eval_aborted_reasons(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[global___EvaluationAbortedReason.ValueType]:
        """always has the same size as the number of intervals"""

    def __init__(
        self,
        *,
        evaluation_started: builtins.bool = ...,
        evaluation_id: builtins.int = ...,
        dataset_sizes: collections.abc.Iterable[builtins.int] | None = ...,
        eval_aborted_reasons: collections.abc.Iterable[global___EvaluationAbortedReason.ValueType] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["dataset_sizes", b"dataset_sizes", "eval_aborted_reasons", b"eval_aborted_reasons", "evaluation_id", b"evaluation_id", "evaluation_started", b"evaluation_started"]) -> None: ...

global___EvaluateModelResponse = EvaluateModelResponse

@typing.final
class EvaluationStatusRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    EVALUATION_ID_FIELD_NUMBER: builtins.int
    evaluation_id: builtins.int
    def __init__(
        self,
        *,
        evaluation_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["evaluation_id", b"evaluation_id"]) -> None: ...

global___EvaluationStatusRequest = EvaluationStatusRequest

@typing.final
class EvaluationStatusResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALID_FIELD_NUMBER: builtins.int
    IS_RUNNING_FIELD_NUMBER: builtins.int
    EXCEPTION_FIELD_NUMBER: builtins.int
    valid: builtins.bool
    is_running: builtins.bool
    exception: builtins.str
    def __init__(
        self,
        *,
        valid: builtins.bool = ...,
        is_running: builtins.bool = ...,
        exception: builtins.str | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["_exception", b"_exception", "exception", b"exception"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["_exception", b"_exception", "exception", b"exception", "is_running", b"is_running", "valid", b"valid"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["_exception", b"_exception"]) -> typing.Literal["exception"] | None: ...

global___EvaluationStatusResponse = EvaluationStatusResponse

@typing.final
class SingleMetricResult(google.protobuf.message.Message):
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
    def ClearField(self, field_name: typing.Literal["metric", b"metric", "result", b"result"]) -> None: ...

global___SingleMetricResult = SingleMetricResult

@typing.final
class SingleEvaluationData(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    INTERVAL_INDEX_FIELD_NUMBER: builtins.int
    EVALUATION_DATA_FIELD_NUMBER: builtins.int
    interval_index: builtins.int
    """multiple metrics are required on on evaluation on one interval"""
    @property
    def evaluation_data(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___SingleMetricResult]: ...
    def __init__(
        self,
        *,
        interval_index: builtins.int = ...,
        evaluation_data: collections.abc.Iterable[global___SingleMetricResult] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["evaluation_data", b"evaluation_data", "interval_index", b"interval_index"]) -> None: ...

global___SingleEvaluationData = SingleEvaluationData

@typing.final
class EvaluationResultRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    EVALUATION_ID_FIELD_NUMBER: builtins.int
    evaluation_id: builtins.int
    def __init__(
        self,
        *,
        evaluation_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["evaluation_id", b"evaluation_id"]) -> None: ...

global___EvaluationResultRequest = EvaluationResultRequest

@typing.final
class EvaluationCleanupRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    EVALUATION_IDS_FIELD_NUMBER: builtins.int
    @property
    def evaluation_ids(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        evaluation_ids: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["evaluation_ids", b"evaluation_ids"]) -> None: ...

global___EvaluationCleanupRequest = EvaluationCleanupRequest

@typing.final
class EvaluationResultResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALID_FIELD_NUMBER: builtins.int
    EVALUATION_RESULTS_FIELD_NUMBER: builtins.int
    valid: builtins.bool
    @property
    def evaluation_results(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___SingleEvaluationData]:
        """each element in the list corresponds to the evaluation results on a single interval"""

    def __init__(
        self,
        *,
        valid: builtins.bool = ...,
        evaluation_results: collections.abc.Iterable[global___SingleEvaluationData] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["evaluation_results", b"evaluation_results", "valid", b"valid"]) -> None: ...

global___EvaluationResultResponse = EvaluationResultResponse

@typing.final
class EvaluationCleanupResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SUCCEEDED_FIELD_NUMBER: builtins.int
    @property
    def succeeded(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        succeeded: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["succeeded", b"succeeded"]) -> None: ...

global___EvaluationCleanupResponse = EvaluationCleanupResponse
