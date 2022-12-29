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

class GetNewDataSinceRequest(_message.Message):
    __slots__ = ["dataset_id", "timestamp"]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    timestamp: int
    def __init__(self, dataset_id: _Optional[str] = ..., timestamp: _Optional[int] = ...) -> None: ...

class GetNewDataSinceResponse(_message.Message):
    __slots__ = ["keys"]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, keys: _Optional[_Iterable[str]] = ...) -> None: ...

class GetRequest(_message.Message):
    __slots__ = ["dataset_id", "keys"]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    dataset_id: str
    keys: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, dataset_id: _Optional[str] = ..., keys: _Optional[_Iterable[str]] = ...) -> None: ...

class GetResponse(_message.Message):
    __slots__ = ["chunk"]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    chunk: bytes
    def __init__(self, chunk: _Optional[bytes] = ...) -> None: ...

class RegisterNewDatasetRequest(_message.Message):
    __slots__ = ["base_path", "dataset_id", "description", "file_wrapper_type", "filesystem_wrapper_type"]
    BASE_PATH_FIELD_NUMBER: _ClassVar[int]
    DATASET_ID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    FILESYSTEM_WRAPPER_TYPE_FIELD_NUMBER: _ClassVar[int]
    FILE_WRAPPER_TYPE_FIELD_NUMBER: _ClassVar[int]
    base_path: str
    dataset_id: str
    description: str
    file_wrapper_type: int
    filesystem_wrapper_type: int
    def __init__(self, dataset_id: _Optional[str] = ..., filesystem_wrapper_type: _Optional[int] = ..., file_wrapper_type: _Optional[int] = ..., description: _Optional[str] = ..., base_path: _Optional[str] = ...) -> None: ...

class RegisterNewDatasetResponse(_message.Message):
    __slots__ = ["success"]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...
