from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GetSamplesRequest(_message.Message):
    __slots__ = ["training_id", "training_set_number", "worker_id"]
    TRAINING_ID_FIELD_NUMBER: _ClassVar[int]
    TRAINING_SET_NUMBER_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    training_id: int
    training_set_number: int
    worker_id: int
    def __init__(self, training_id: _Optional[int] = ..., training_set_number: _Optional[int] = ..., worker_id: _Optional[int] = ...) -> None: ...

class RegisterTrainingRequest(_message.Message):
    __slots__ = ["num_workers", "training_set_size"]
    NUM_WORKERS_FIELD_NUMBER: _ClassVar[int]
    TRAINING_SET_SIZE_FIELD_NUMBER: _ClassVar[int]
    num_workers: int
    training_set_size: int
    def __init__(self, training_set_size: _Optional[int] = ..., num_workers: _Optional[int] = ...) -> None: ...

class SamplesResponse(_message.Message):
    __slots__ = ["training_samples_subset"]
    TRAINING_SAMPLES_SUBSET_FIELD_NUMBER: _ClassVar[int]
    training_samples_subset: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, training_samples_subset: _Optional[_Iterable[str]] = ...) -> None: ...

class TrainingResponse(_message.Message):
    __slots__ = ["training_id"]
    TRAINING_ID_FIELD_NUMBER: _ClassVar[int]
    training_id: int
    def __init__(self, training_id: _Optional[int] = ...) -> None: ...
