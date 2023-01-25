from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class DataInformRequest(_message.Message):
    __slots__ = ["keys", "labels", "pipeline_id", "timestamps"]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedScalarFieldContainer[str]
    labels: _containers.RepeatedScalarFieldContainer[int]
    pipeline_id: int
    timestamps: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, pipeline_id: _Optional[int] = ..., keys: _Optional[_Iterable[str]] = ..., timestamps: _Optional[_Iterable[int]] = ..., labels: _Optional[_Iterable[int]] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetSamplesRequest(_message.Message):
    __slots__ = ["pipeline_id", "trigger_id", "worker_id"]
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    pipeline_id: int
    trigger_id: int
    worker_id: int
    def __init__(self, pipeline_id: _Optional[int] = ..., trigger_id: _Optional[int] = ..., worker_id: _Optional[int] = ...) -> None: ...

class PipelineResponse(_message.Message):
    __slots__ = ["pipeline_id"]
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    pipeline_id: int
    def __init__(self, pipeline_id: _Optional[int] = ...) -> None: ...

class RegisterPipelineRequest(_message.Message):
    __slots__ = ["num_workers", "selector_strategy_config"]
    NUM_WORKERS_FIELD_NUMBER: _ClassVar[int]
    SELECTOR_STRATEGY_CONFIG_FIELD_NUMBER: _ClassVar[int]
    num_workers: int
    selector_strategy_config: str
    def __init__(self, num_workers: _Optional[int] = ..., selector_strategy_config: _Optional[str] = ...) -> None: ...

class SamplesResponse(_message.Message):
    __slots__ = ["training_samples_subset", "training_samples_weight"]
    TRAINING_SAMPLES_SUBSET_FIELD_NUMBER: _ClassVar[int]
    TRAINING_SAMPLES_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    training_samples_subset: _containers.RepeatedScalarFieldContainer[str]
    training_samples_weight: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, training_samples_subset: _Optional[_Iterable[str]] = ..., training_samples_weight: _Optional[_Iterable[float]] = ...) -> None: ...

class TriggerResponse(_message.Message):
    __slots__ = ["trigger_id"]
    TRIGGER_ID_FIELD_NUMBER: _ClassVar[int]
    trigger_id: int
    def __init__(self, trigger_id: _Optional[int] = ...) -> None: ...
