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
    __slots__ = ["data", "keys", "label", "scores", "seen"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    SEEN_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[str]
    keys: _containers.RepeatedScalarFieldContainer[str]
    label: _containers.RepeatedScalarFieldContainer[int]
    scores: _containers.RepeatedScalarFieldContainer[float]
    seen: _containers.RepeatedScalarFieldContainer[bool]
    def __init__(self, keys: _Optional[_Iterable[str]] = ..., scores: _Optional[_Iterable[float]] = ..., data: _Optional[_Iterable[str]] = ..., seen: _Optional[_Iterable[bool]] = ..., label: _Optional[_Iterable[int]] = ...) -> None: ...

class GetTrainingRequest(_message.Message):
    __slots__ = ["training_id"]
    TRAINING_ID_FIELD_NUMBER: _ClassVar[int]
    training_id: int
    def __init__(self, training_id: _Optional[int] = ...) -> None: ...

class RegisterRequest(_message.Message):
    __slots__ = ["num_workers", "training_id", "training_set_size"]
    NUM_WORKERS_FIELD_NUMBER: _ClassVar[int]
    TRAINING_ID_FIELD_NUMBER: _ClassVar[int]
    TRAINING_SET_SIZE_FIELD_NUMBER: _ClassVar[int]
    num_workers: int
    training_id: int
    training_set_size: int
    def __init__(self, training_id: _Optional[int] = ..., training_set_size: _Optional[int] = ..., num_workers: _Optional[int] = ...) -> None: ...

class RegisterResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

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

class TrainingResponse(_message.Message):
    __slots__ = ["num_workers", "training_set_size"]
    NUM_WORKERS_FIELD_NUMBER: _ClassVar[int]
    TRAINING_SET_SIZE_FIELD_NUMBER: _ClassVar[int]
    num_workers: int
    training_set_size: int
    def __init__(self, training_set_size: _Optional[int] = ..., num_workers: _Optional[int] = ...) -> None: ...
