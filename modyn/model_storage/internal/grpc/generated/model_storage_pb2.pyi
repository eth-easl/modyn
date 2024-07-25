"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""

import builtins
import google.protobuf.descriptor
import google.protobuf.message
import typing

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing.final
class RegisterModelRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_ID_FIELD_NUMBER: builtins.int
    TRIGGER_ID_FIELD_NUMBER: builtins.int
    HOSTNAME_FIELD_NUMBER: builtins.int
    PORT_FIELD_NUMBER: builtins.int
    MODEL_PATH_FIELD_NUMBER: builtins.int
    CHECKSUM_FIELD_NUMBER: builtins.int
    pipeline_id: builtins.int
    trigger_id: builtins.int
    hostname: builtins.str
    port: builtins.int
    model_path: builtins.str
    checksum: builtins.bytes
    def __init__(
        self,
        *,
        pipeline_id: builtins.int = ...,
        trigger_id: builtins.int = ...,
        hostname: builtins.str = ...,
        port: builtins.int = ...,
        model_path: builtins.str = ...,
        checksum: builtins.bytes = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["checksum", b"checksum", "hostname", b"hostname", "model_path", b"model_path", "pipeline_id", b"pipeline_id", "port", b"port", "trigger_id", b"trigger_id"]) -> None: ...

global___RegisterModelRequest = RegisterModelRequest

@typing.final
class RegisterModelResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SUCCESS_FIELD_NUMBER: builtins.int
    MODEL_ID_FIELD_NUMBER: builtins.int
    success: builtins.bool
    model_id: builtins.int
    def __init__(
        self,
        *,
        success: builtins.bool = ...,
        model_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["model_id", b"model_id", "success", b"success"]) -> None: ...

global___RegisterModelResponse = RegisterModelResponse

@typing.final
class FetchModelRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    MODEL_ID_FIELD_NUMBER: builtins.int
    LOAD_METADATA_FIELD_NUMBER: builtins.int
    DEVICE_FIELD_NUMBER: builtins.int
    model_id: builtins.int
    load_metadata: builtins.bool
    device: builtins.str
    def __init__(
        self,
        *,
        model_id: builtins.int = ...,
        load_metadata: builtins.bool = ...,
        device: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["device", b"device", "load_metadata", b"load_metadata", "model_id", b"model_id"]) -> None: ...

global___FetchModelRequest = FetchModelRequest

@typing.final
class FetchModelResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SUCCESS_FIELD_NUMBER: builtins.int
    MODEL_PATH_FIELD_NUMBER: builtins.int
    CHECKSUM_FIELD_NUMBER: builtins.int
    success: builtins.bool
    model_path: builtins.str
    checksum: builtins.bytes
    def __init__(
        self,
        *,
        success: builtins.bool = ...,
        model_path: builtins.str = ...,
        checksum: builtins.bytes = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["checksum", b"checksum", "model_path", b"model_path", "success", b"success"]) -> None: ...

global___FetchModelResponse = FetchModelResponse

@typing.final
class DeleteModelRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    MODEL_ID_FIELD_NUMBER: builtins.int
    model_id: builtins.int
    def __init__(
        self,
        *,
        model_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["model_id", b"model_id"]) -> None: ...

global___DeleteModelRequest = DeleteModelRequest

@typing.final
class DeleteModelResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SUCCESS_FIELD_NUMBER: builtins.int
    success: builtins.bool
    def __init__(
        self,
        *,
        success: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["success", b"success"]) -> None: ...

global___DeleteModelResponse = DeleteModelResponse
