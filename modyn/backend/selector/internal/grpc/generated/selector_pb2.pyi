from typing import ClassVar as _ClassVar
from typing import Iterable as _Iterable
from typing import Optional as _Optional

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

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
    __slots__ = ["num_workers"]
    NUM_WORKERS_FIELD_NUMBER: _ClassVar[int]
    num_workers: int
    def __init__(self, num_workers: _Optional[int] = ...) -> None: ...

class SamplesResponse(_message.Message):
    __slots__ = ["training_samples_subset", "training_samples_weight"]
    TRAINING_SAMPLES_SUBSET_FIELD_NUMBER: _ClassVar[int]
    TRAINING_SAMPLES_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    training_samples_subset: _containers.RepeatedScalarFieldContainer[str]
    training_samples_weight: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, training_samples_subset: _Optional[_Iterable[str]] = ..., training_samples_weight: _Optional[_Iterable[float]] = ...) -> None: ...

class TrainingResponse(_message.Message):
    __slots__ = ["training_id"]
    TRAINING_ID_FIELD_NUMBER: _ClassVar[int]
    training_id: int
    def __init__(self, training_id: _Optional[int] = ...) -> None: ...
