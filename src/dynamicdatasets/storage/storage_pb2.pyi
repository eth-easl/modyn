from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GetRequest(_message.Message):
    __slots__ = ["keys"]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, keys: _Optional[_Iterable[str]] = ...) -> None: ...

class GetResponse(_message.Message):
    __slots__ = ["value"]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, value: _Optional[_Iterable[bytes]] = ...) -> None: ...

class PutRequest(_message.Message):
    __slots__ = ["keys", "value"]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedScalarFieldContainer[str]
    value: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, keys: _Optional[_Iterable[str]] = ..., value: _Optional[_Iterable[bytes]] = ...) -> None: ...

class PutResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...
