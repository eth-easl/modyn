from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class DeleteRequest(_message.Message):
    __slots__ = ["training_id"]
    TRAINING_ID_FIELD_NUMBER: _ClassVar[int]
    training_id: int
    def __init__(self, training_id: _Optional[int] = ...) -> None: ...

class DeleteResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetByKeysRequest(_message.Message):
    __slots__ = ["keys", "training_id"]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    TRAINING_ID_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedScalarFieldContainer[str]
    training_id: int
    def __init__(self, training_id: _Optional[int] = ..., keys: _Optional[_Iterable[str]] = ...) -> None: ...

class GetByQueryRequest(_message.Message):
    __slots__ = ["query"]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    query: str
    def __init__(self, query: _Optional[str] = ...) -> None: ...

class GetKeysResponse(_message.Message):
    __slots__ = ["keys"]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, keys: _Optional[_Iterable[str]] = ...) -> None: ...

class GetResponse(_message.Message):
    __slots__ = ["data", "keys", "scores"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[str]
    keys: _containers.RepeatedScalarFieldContainer[str]
    scores: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, keys: _Optional[_Iterable[str]] = ..., scores: _Optional[_Iterable[float]] = ..., data: _Optional[_Iterable[str]] = ...) -> None: ...

class SetRequest(_message.Message):
    __slots__ = ["data", "keys", "scores", "training_id"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    TRAINING_ID_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[str]
    keys: _containers.RepeatedScalarFieldContainer[str]
    scores: _containers.RepeatedScalarFieldContainer[float]
    training_id: int
    def __init__(self, training_id: _Optional[int] = ..., keys: _Optional[_Iterable[str]] = ..., scores: _Optional[_Iterable[float]] = ..., data: _Optional[_Iterable[str]] = ...) -> None: ...

class SetResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...
