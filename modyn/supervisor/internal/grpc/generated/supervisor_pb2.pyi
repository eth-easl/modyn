"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""

import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing.final
class JsonString(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALUE_FIELD_NUMBER: builtins.int
    value: builtins.str
    def __init__(
        self,
        *,
        value: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["value", b"value"]) -> None: ...

global___JsonString = JsonString

@typing.final
class StartPipelineRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_CONFIG_FIELD_NUMBER: builtins.int
    START_REPLAY_AT_FIELD_NUMBER: builtins.int
    STOP_REPLAY_AT_FIELD_NUMBER: builtins.int
    MAXIMUM_TRIGGERS_FIELD_NUMBER: builtins.int
    start_replay_at: builtins.int
    stop_replay_at: builtins.int
    maximum_triggers: builtins.int
    @property
    def pipeline_config(self) -> global___JsonString: ...
    def __init__(
        self,
        *,
        pipeline_config: global___JsonString | None = ...,
        start_replay_at: builtins.int | None = ...,
        stop_replay_at: builtins.int | None = ...,
        maximum_triggers: builtins.int | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["_maximum_triggers", b"_maximum_triggers", "_start_replay_at", b"_start_replay_at", "_stop_replay_at", b"_stop_replay_at", "maximum_triggers", b"maximum_triggers", "pipeline_config", b"pipeline_config", "start_replay_at", b"start_replay_at", "stop_replay_at", b"stop_replay_at"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["_maximum_triggers", b"_maximum_triggers", "_start_replay_at", b"_start_replay_at", "_stop_replay_at", b"_stop_replay_at", "maximum_triggers", b"maximum_triggers", "pipeline_config", b"pipeline_config", "start_replay_at", b"start_replay_at", "stop_replay_at", b"stop_replay_at"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing.Literal["_maximum_triggers", b"_maximum_triggers"]) -> typing.Literal["maximum_triggers"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing.Literal["_start_replay_at", b"_start_replay_at"]) -> typing.Literal["start_replay_at"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing.Literal["_stop_replay_at", b"_stop_replay_at"]) -> typing.Literal["stop_replay_at"] | None: ...

global___StartPipelineRequest = StartPipelineRequest

@typing.final
class PipelineResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_ID_FIELD_NUMBER: builtins.int
    EXCEPTION_FIELD_NUMBER: builtins.int
    pipeline_id: builtins.int
    exception: builtins.str
    def __init__(
        self,
        *,
        pipeline_id: builtins.int = ...,
        exception: builtins.str | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["_exception", b"_exception", "exception", b"exception"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["_exception", b"_exception", "exception", b"exception", "pipeline_id", b"pipeline_id"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["_exception", b"_exception"]) -> typing.Literal["exception"] | None: ...

global___PipelineResponse = PipelineResponse

@typing.final
class GetPipelineStatusRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_ID_FIELD_NUMBER: builtins.int
    pipeline_id: builtins.int
    def __init__(
        self,
        *,
        pipeline_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["pipeline_id", b"pipeline_id"]) -> None: ...

global___GetPipelineStatusRequest = GetPipelineStatusRequest

@typing.final
class PipelineStageIdMsg(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ID_TYPE_FIELD_NUMBER: builtins.int
    ID_FIELD_NUMBER: builtins.int
    id_type: builtins.str
    id: builtins.int
    def __init__(
        self,
        *,
        id_type: builtins.str = ...,
        id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["id", b"id", "id_type", b"id_type"]) -> None: ...

global___PipelineStageIdMsg = PipelineStageIdMsg

@typing.final
class PipelineStageDatasetMsg(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ID_FIELD_NUMBER: builtins.int
    id: builtins.str
    def __init__(
        self,
        *,
        id: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["id", b"id"]) -> None: ...

global___PipelineStageDatasetMsg = PipelineStageDatasetMsg

@typing.final
class PipelineStageCounterCreateParams(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TITLE_FIELD_NUMBER: builtins.int
    NEW_DATA_LEN_FIELD_NUMBER: builtins.int
    title: builtins.str
    new_data_len: builtins.int
    def __init__(
        self,
        *,
        title: builtins.str = ...,
        new_data_len: builtins.int | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["_new_data_len", b"_new_data_len", "new_data_len", b"new_data_len"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["_new_data_len", b"_new_data_len", "new_data_len", b"new_data_len", "title", b"title"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["_new_data_len", b"_new_data_len"]) -> typing.Literal["new_data_len"] | None: ...

global___PipelineStageCounterCreateParams = PipelineStageCounterCreateParams

@typing.final
class PipelineStageCounterUpdateParams(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    INCREMENT_FIELD_NUMBER: builtins.int
    increment: builtins.int
    def __init__(
        self,
        *,
        increment: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["increment", b"increment"]) -> None: ...

global___PipelineStageCounterUpdateParams = PipelineStageCounterUpdateParams

@typing.final
class PipelineStageCounterMsg(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ACTION_FIELD_NUMBER: builtins.int
    CREATE_PARAMS_FIELD_NUMBER: builtins.int
    UPDATE_PARAMS_FIELD_NUMBER: builtins.int
    action: builtins.str
    @property
    def create_params(self) -> global___PipelineStageCounterCreateParams: ...
    @property
    def update_params(self) -> global___PipelineStageCounterUpdateParams: ...
    def __init__(
        self,
        *,
        action: builtins.str = ...,
        create_params: global___PipelineStageCounterCreateParams | None = ...,
        update_params: global___PipelineStageCounterUpdateParams | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["create_params", b"create_params", "params", b"params", "update_params", b"update_params"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["action", b"action", "create_params", b"create_params", "params", b"params", "update_params", b"update_params"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["params", b"params"]) -> typing.Literal["create_params", "update_params"] | None: ...

global___PipelineStageCounterMsg = PipelineStageCounterMsg

@typing.final
class PipelineStageExitMsg(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    EXITCODE_FIELD_NUMBER: builtins.int
    EXCEPTION_FIELD_NUMBER: builtins.int
    exitcode: builtins.int
    exception: builtins.str
    def __init__(
        self,
        *,
        exitcode: builtins.int = ...,
        exception: builtins.str | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["_exception", b"_exception", "exception", b"exception"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["_exception", b"_exception", "exception", b"exception", "exitcode", b"exitcode"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["_exception", b"_exception"]) -> typing.Literal["exception"] | None: ...

global___PipelineStageExitMsg = PipelineStageExitMsg

@typing.final
class PipelineStage(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STAGE_FIELD_NUMBER: builtins.int
    MSG_TYPE_FIELD_NUMBER: builtins.int
    LOG_FIELD_NUMBER: builtins.int
    ID_MSG_FIELD_NUMBER: builtins.int
    DATASET_MSG_FIELD_NUMBER: builtins.int
    COUNTER_MSG_FIELD_NUMBER: builtins.int
    EXIT_MSG_FIELD_NUMBER: builtins.int
    stage: builtins.str
    msg_type: builtins.str
    log: builtins.bool
    @property
    def id_msg(self) -> global___PipelineStageIdMsg: ...
    @property
    def dataset_msg(self) -> global___PipelineStageDatasetMsg: ...
    @property
    def counter_msg(self) -> global___PipelineStageCounterMsg: ...
    @property
    def exit_msg(self) -> global___PipelineStageExitMsg: ...
    def __init__(
        self,
        *,
        stage: builtins.str = ...,
        msg_type: builtins.str = ...,
        log: builtins.bool = ...,
        id_msg: global___PipelineStageIdMsg | None = ...,
        dataset_msg: global___PipelineStageDatasetMsg | None = ...,
        counter_msg: global___PipelineStageCounterMsg | None = ...,
        exit_msg: global___PipelineStageExitMsg | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["counter_msg", b"counter_msg", "dataset_msg", b"dataset_msg", "exit_msg", b"exit_msg", "id_msg", b"id_msg", "msg", b"msg"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["counter_msg", b"counter_msg", "dataset_msg", b"dataset_msg", "exit_msg", b"exit_msg", "id_msg", b"id_msg", "log", b"log", "msg", b"msg", "msg_type", b"msg_type", "stage", b"stage"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["msg", b"msg"]) -> typing.Literal["id_msg", "dataset_msg", "counter_msg", "exit_msg"] | None: ...

global___PipelineStage = PipelineStage

@typing.final
class TrainingStatusCreateTrackerParams(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TOTAL_SAMPLES_FIELD_NUMBER: builtins.int
    STATUS_BAR_SCALE_FIELD_NUMBER: builtins.int
    total_samples: builtins.int
    status_bar_scale: builtins.int
    def __init__(
        self,
        *,
        total_samples: builtins.int = ...,
        status_bar_scale: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["status_bar_scale", b"status_bar_scale", "total_samples", b"total_samples"]) -> None: ...

global___TrainingStatusCreateTrackerParams = TrainingStatusCreateTrackerParams

@typing.final
class TrainingStatusProgressCounterParams(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SAMPLES_SEEN_FIELD_NUMBER: builtins.int
    DOWNSAMPLING_SAMPLES_SEEN_FIELD_NUMBER: builtins.int
    IS_TRAINING_FIELD_NUMBER: builtins.int
    samples_seen: builtins.int
    downsampling_samples_seen: builtins.int
    is_training: builtins.bool
    def __init__(
        self,
        *,
        samples_seen: builtins.int = ...,
        downsampling_samples_seen: builtins.int = ...,
        is_training: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["downsampling_samples_seen", b"downsampling_samples_seen", "is_training", b"is_training", "samples_seen", b"samples_seen"]) -> None: ...

global___TrainingStatusProgressCounterParams = TrainingStatusProgressCounterParams

@typing.final
class TrainingStatus(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STAGE_FIELD_NUMBER: builtins.int
    ACTION_FIELD_NUMBER: builtins.int
    ID_FIELD_NUMBER: builtins.int
    TRAINING_CREATE_TRACKER_PARAMS_FIELD_NUMBER: builtins.int
    TRAINING_PROGRESS_COUNTER_PARAMS_FIELD_NUMBER: builtins.int
    stage: builtins.str
    action: builtins.str
    id: builtins.int
    @property
    def training_create_tracker_params(self) -> global___TrainingStatusCreateTrackerParams: ...
    @property
    def training_progress_counter_params(self) -> global___TrainingStatusProgressCounterParams: ...
    def __init__(
        self,
        *,
        stage: builtins.str = ...,
        action: builtins.str = ...,
        id: builtins.int = ...,
        training_create_tracker_params: global___TrainingStatusCreateTrackerParams | None = ...,
        training_progress_counter_params: global___TrainingStatusProgressCounterParams | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["params", b"params", "training_create_tracker_params", b"training_create_tracker_params", "training_progress_counter_params", b"training_progress_counter_params"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["action", b"action", "id", b"id", "params", b"params", "stage", b"stage", "training_create_tracker_params", b"training_create_tracker_params", "training_progress_counter_params", b"training_progress_counter_params"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["params", b"params"]) -> typing.Literal["training_create_tracker_params", "training_progress_counter_params"] | None: ...

global___TrainingStatus = TrainingStatus

@typing.final
class EvalStatusCreateTrackerParams(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATASET_ID_FIELD_NUMBER: builtins.int
    DATASET_SIZE_FIELD_NUMBER: builtins.int
    dataset_id: builtins.str
    dataset_size: builtins.int
    def __init__(
        self,
        *,
        dataset_id: builtins.str = ...,
        dataset_size: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["dataset_id", b"dataset_id", "dataset_size", b"dataset_size"]) -> None: ...

global___EvalStatusCreateTrackerParams = EvalStatusCreateTrackerParams

@typing.final
class EvalStatusCreateCounterParams(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRAINING_ID_FIELD_NUMBER: builtins.int
    training_id: builtins.int
    def __init__(
        self,
        *,
        training_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["training_id", b"training_id"]) -> None: ...

global___EvalStatusCreateCounterParams = EvalStatusCreateCounterParams

@typing.final
class EvalStatusProgressCounterParams(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TOTAL_SAMPLES_SEEN_FIELD_NUMBER: builtins.int
    total_samples_seen: builtins.int
    def __init__(
        self,
        *,
        total_samples_seen: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["total_samples_seen", b"total_samples_seen"]) -> None: ...

global___EvalStatusProgressCounterParams = EvalStatusProgressCounterParams

@typing.final
class EvalStatusEndCounterParams(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ERROR_FIELD_NUMBER: builtins.int
    EXCEPTION_MSG_FIELD_NUMBER: builtins.int
    error: builtins.bool
    exception_msg: builtins.str
    def __init__(
        self,
        *,
        error: builtins.bool = ...,
        exception_msg: builtins.str | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["_exception_msg", b"_exception_msg", "exception_msg", b"exception_msg"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["_exception_msg", b"_exception_msg", "error", b"error", "exception_msg", b"exception_msg"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["_exception_msg", b"_exception_msg"]) -> typing.Literal["exception_msg"] | None: ...

global___EvalStatusEndCounterParams = EvalStatusEndCounterParams

@typing.final
class EvalStatus(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STAGE_FIELD_NUMBER: builtins.int
    ACTION_FIELD_NUMBER: builtins.int
    ID_FIELD_NUMBER: builtins.int
    EVAL_CREATE_TRACKER_PARAMS_FIELD_NUMBER: builtins.int
    EVAL_CREATE_COUNTER_PARAMS_FIELD_NUMBER: builtins.int
    EVAL_PROGRESS_COUNTER_PARAMS_FIELD_NUMBER: builtins.int
    EVAL_END_COUNTER_PARAMS_FIELD_NUMBER: builtins.int
    stage: builtins.str
    action: builtins.str
    id: builtins.int
    @property
    def eval_create_tracker_params(self) -> global___EvalStatusCreateTrackerParams: ...
    @property
    def eval_create_counter_params(self) -> global___EvalStatusCreateCounterParams: ...
    @property
    def eval_progress_counter_params(self) -> global___EvalStatusProgressCounterParams: ...
    @property
    def eval_end_counter_params(self) -> global___EvalStatusEndCounterParams: ...
    def __init__(
        self,
        *,
        stage: builtins.str = ...,
        action: builtins.str = ...,
        id: builtins.int = ...,
        eval_create_tracker_params: global___EvalStatusCreateTrackerParams | None = ...,
        eval_create_counter_params: global___EvalStatusCreateCounterParams | None = ...,
        eval_progress_counter_params: global___EvalStatusProgressCounterParams | None = ...,
        eval_end_counter_params: global___EvalStatusEndCounterParams | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["eval_create_counter_params", b"eval_create_counter_params", "eval_create_tracker_params", b"eval_create_tracker_params", "eval_end_counter_params", b"eval_end_counter_params", "eval_progress_counter_params", b"eval_progress_counter_params", "params", b"params"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["action", b"action", "eval_create_counter_params", b"eval_create_counter_params", "eval_create_tracker_params", b"eval_create_tracker_params", "eval_end_counter_params", b"eval_end_counter_params", "eval_progress_counter_params", b"eval_progress_counter_params", "id", b"id", "params", b"params", "stage", b"stage"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["params", b"params"]) -> typing.Literal["eval_create_tracker_params", "eval_create_counter_params", "eval_progress_counter_params", "eval_end_counter_params"] | None: ...

global___EvalStatus = EvalStatus

@typing.final
class GetPipelineStatusResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STATUS_FIELD_NUMBER: builtins.int
    PIPELINE_STAGE_FIELD_NUMBER: builtins.int
    TRAINING_STATUS_FIELD_NUMBER: builtins.int
    EVAL_STATUS_FIELD_NUMBER: builtins.int
    status: builtins.str
    @property
    def pipeline_stage(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___PipelineStage]: ...
    @property
    def training_status(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___TrainingStatus]: ...
    @property
    def eval_status(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___EvalStatus]: ...
    def __init__(
        self,
        *,
        status: builtins.str = ...,
        pipeline_stage: collections.abc.Iterable[global___PipelineStage] | None = ...,
        training_status: collections.abc.Iterable[global___TrainingStatus] | None = ...,
        eval_status: collections.abc.Iterable[global___EvalStatus] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["eval_status", b"eval_status", "pipeline_stage", b"pipeline_stage", "status", b"status", "training_status", b"training_status"]) -> None: ...

global___GetPipelineStatusResponse = GetPipelineStatusResponse
