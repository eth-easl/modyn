from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class AddRequest(_message.Message):
    __slots__ = ["keys"]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, keys: _Optional[_Iterable[str]] = ...) -> None: ...

class AddResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetNextRequest(_message.Message):
    __slots__ = ["limit", "training_id"]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    TRAINING_ID_FIELD_NUMBER: _ClassVar[int]
    limit: int
    training_id: int
    def __init__(self, limit: _Optional[int] = ..., training_id: _Optional[int] = ...) -> None: ...

class GetNextResponse(_message.Message):
    __slots__ = ["keys"]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, keys: _Optional[_Iterable[str]] = ...) -> None: ...
