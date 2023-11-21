"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.message
import sys
import typing

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class JsonString(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALUE_FIELD_NUMBER: builtins.int
    value: builtins.str
    def __init__(
        self,
        *,
        value: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["value", b"value"]) -> None: ...

global___JsonString = JsonString

@typing_extensions.final
class StartPipelineRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_CONFIG_FIELD_NUMBER: builtins.int
    EVAL_DIRECTORY_FIELD_NUMBER: builtins.int
    START_REPLAY_AT_FIELD_NUMBER: builtins.int
    STOP_REPLAY_AT_FIELD_NUMBER: builtins.int
    MAXIMUM_TRIGGERS_FIELD_NUMBER: builtins.int
    @property
    def pipeline_config(self) -> global___JsonString: ...
    eval_directory: builtins.str
    start_replay_at: builtins.int
    stop_replay_at: builtins.int
    maximum_triggers: builtins.int
    def __init__(
        self,
        *,
        pipeline_config: global___JsonString | None = ...,
        eval_directory: builtins.str = ...,
        start_replay_at: builtins.int | None = ...,
        stop_replay_at: builtins.int | None = ...,
        maximum_triggers: builtins.int | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_maximum_triggers", b"_maximum_triggers", "_start_replay_at", b"_start_replay_at", "_stop_replay_at", b"_stop_replay_at", "maximum_triggers", b"maximum_triggers", "pipeline_config", b"pipeline_config", "start_replay_at", b"start_replay_at", "stop_replay_at", b"stop_replay_at"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_maximum_triggers", b"_maximum_triggers", "_start_replay_at", b"_start_replay_at", "_stop_replay_at", b"_stop_replay_at", "eval_directory", b"eval_directory", "maximum_triggers", b"maximum_triggers", "pipeline_config", b"pipeline_config", "start_replay_at", b"start_replay_at", "stop_replay_at", b"stop_replay_at"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_maximum_triggers", b"_maximum_triggers"]) -> typing_extensions.Literal["maximum_triggers"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_start_replay_at", b"_start_replay_at"]) -> typing_extensions.Literal["start_replay_at"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_stop_replay_at", b"_stop_replay_at"]) -> typing_extensions.Literal["stop_replay_at"] | None: ...

global___StartPipelineRequest = StartPipelineRequest

@typing_extensions.final
class PipelineResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_ID_FIELD_NUMBER: builtins.int
    pipeline_id: builtins.int
    def __init__(
        self,
        *,
        pipeline_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["pipeline_id", b"pipeline_id"]) -> None: ...

global___PipelineResponse = PipelineResponse

@typing_extensions.final
class GetPipelineStatusRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_ID_FIELD_NUMBER: builtins.int
    pipeline_id: builtins.int
    def __init__(
        self,
        *,
        pipeline_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["pipeline_id", b"pipeline_id"]) -> None: ...

global___GetPipelineStatusRequest = GetPipelineStatusRequest

@typing_extensions.final
class GetPipelineStatusResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STATUS_FIELD_NUMBER: builtins.int
    DETAIL_FIELD_NUMBER: builtins.int
    status: builtins.str
    @property
    def detail(self) -> global___JsonString: ...
    def __init__(
        self,
        *,
        status: builtins.str = ...,
        detail: global___JsonString | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["detail", b"detail"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["detail", b"detail", "status", b"status"]) -> None: ...

global___GetPipelineStatusResponse = GetPipelineStatusResponse
