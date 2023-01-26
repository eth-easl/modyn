from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class DatasetAvailableRequest(_message.Message):
    __slots__ = ["dataset_id"]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    def __init__(self, dataset_id: _Optional[str] = ...) -> None: ...

class DatasetAvailableResponse(_message.Message):
    __slots__ = ["available"]
    AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    available: bool
    def __init__(self, available: bool = ...) -> None: ...

class DeleteDatasetResponse(_message.Message):
    __slots__ = ["success"]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class GetCurrentTimestampResponse(_message.Message):
    __slots__ = ["timestamp"]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    timestamp: int
    def __init__(self, timestamp: _Optional[int] = ...) -> None: ...

class GetDataInIntervalRequest(_message.Message):
    __slots__ = ["dataset_id", "end_timestamp", "start_timestamp"]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    END_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    START_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    end_timestamp: int
    start_timestamp: int
    def __init__(self, dataset_id: _Optional[str] = ..., start_timestamp: _Optional[int] = ..., end_timestamp: _Optional[int] = ...) -> None: ...

class GetDataInIntervalResponse(_message.Message):
    __slots__ = ["keys", "timestamps"]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedScalarFieldContainer[str]
    timestamps: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, keys: _Optional[_Iterable[str]] = ..., timestamps: _Optional[_Iterable[int]] = ...) -> None: ...

class GetNewDataSinceRequest(_message.Message):
    __slots__ = ["dataset_id", "timestamp"]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    timestamp: int
    def __init__(self, dataset_id: _Optional[str] = ..., timestamp: _Optional[int] = ...) -> None: ...

class GetNewDataSinceResponse(_message.Message):
    __slots__ = ["keys", "timestamps"]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedScalarFieldContainer[str]
    timestamps: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, keys: _Optional[_Iterable[str]] = ..., timestamps: _Optional[_Iterable[int]] = ...) -> None: ...

class GetRequest(_message.Message):
    __slots__ = ["dataset_id", "keys"]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    keys: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, dataset_id: _Optional[str] = ..., keys: _Optional[_Iterable[str]] = ...) -> None: ...

class GetResponse(_message.Message):
    __slots__ = ["keys", "labels", "samples"]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedScalarFieldContainer[str]
    labels: _containers.RepeatedScalarFieldContainer[int]
    samples: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, samples: _Optional[_Iterable[bytes]] = ..., keys: _Optional[_Iterable[str]] = ..., labels: _Optional[_Iterable[int]] = ...) -> None: ...

class RegisterNewDatasetRequest(_message.Message):
    __slots__ = ["base_path", "dataset_id", "description", "file_wrapper_config", "file_wrapper_type", "filesystem_wrapper_type", "version"]
    BASE_PATH_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    FILESYSTEM_WRAPPER_TYPE_FIELD_NUMBER: _ClassVar[int]
    FILE_WRAPPER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    FILE_WRAPPER_TYPE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    base_path: str
    dataset_id: str
    description: str
    file_wrapper_config: str
    file_wrapper_type: str
    filesystem_wrapper_type: str
    version: str
    def __init__(self, dataset_id: _Optional[str] = ..., filesystem_wrapper_type: _Optional[str] = ..., file_wrapper_type: _Optional[str] = ..., description: _Optional[str] = ..., base_path: _Optional[str] = ..., version: _Optional[str] = ..., file_wrapper_config: _Optional[str] = ...) -> None: ...

class RegisterNewDatasetResponse(_message.Message):
    __slots__ = ["success"]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...
