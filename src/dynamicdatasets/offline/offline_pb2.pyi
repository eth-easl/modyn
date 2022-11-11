from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataRequest(_message.Message):
    __slots__ = ["dataMap"]
    DATAMAP_FIELD_NUMBER: _ClassVar[int]
    dataMap: _struct_pb2.Struct
    def __init__(self, dataMap: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class DataResponse(_message.Message):
    __slots__ = ["dataMap"]
    DATAMAP_FIELD_NUMBER: _ClassVar[int]
    dataMap: _struct_pb2.Struct
    def __init__(self, dataMap: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...
