from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PostTrainingMetadataRequest(_message.Message):
    __slots__ = ["data", "training_id"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    TRAINING_ID_FIELD_NUMBER: _ClassVar[int]
    data: str
    training_id: int
    def __init__(self, training_id: _Optional[int] = ..., data: _Optional[str] = ...) -> None: ...

class PostTrainingMetadataResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...
