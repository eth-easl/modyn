from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Batch(_message.Message):
    __slots__ = ["filename", "rows"]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    filename: str
    rows: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, filename: _Optional[str] = ..., rows: _Optional[_Iterable[int]] = ...) -> None: ...

class BatchData(_message.Message):
    __slots__ = ["dataMap"]
    DATAMAP_FIELD_NUMBER: _ClassVar[int]
    dataMap: _struct_pb2.Struct
    def __init__(self, dataMap: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class BatchId(_message.Message):
    __slots__ = ["batchId"]
    BATCHID_FIELD_NUMBER: _ClassVar[int]
    batchId: int
    def __init__(self, batchId: _Optional[int] = ...) -> None: ...
