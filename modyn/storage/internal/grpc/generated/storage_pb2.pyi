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
class GetRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATASET_ID_FIELD_NUMBER: builtins.int
    KEYS_FIELD_NUMBER: builtins.int
    dataset_id: builtins.str
    @property
    def keys(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        dataset_id: builtins.str = ...,
        keys: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["dataset_id", b"dataset_id", "keys", b"keys"]) -> None: ...

global___GetRequest = GetRequest

@typing.final
class GetResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SAMPLES_FIELD_NUMBER: builtins.int
    KEYS_FIELD_NUMBER: builtins.int
    LABELS_FIELD_NUMBER: builtins.int
    @property
    def samples(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.bytes]: ...
    @property
    def keys(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    @property
    def labels(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        samples: collections.abc.Iterable[builtins.bytes] | None = ...,
        keys: collections.abc.Iterable[builtins.int] | None = ...,
        labels: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["keys", b"keys", "labels", b"labels", "samples", b"samples"]) -> None: ...

global___GetResponse = GetResponse

@typing.final
class GetCurrentTimestampRequest(google.protobuf.message.Message):
    """https://github.com/grpc/grpc/issues/15937"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___GetCurrentTimestampRequest = GetCurrentTimestampRequest

@typing.final
class GetNewDataSinceRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATASET_ID_FIELD_NUMBER: builtins.int
    TIMESTAMP_FIELD_NUMBER: builtins.int
    dataset_id: builtins.str
    timestamp: builtins.int
    def __init__(
        self,
        *,
        dataset_id: builtins.str = ...,
        timestamp: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["dataset_id", b"dataset_id", "timestamp", b"timestamp"]) -> None: ...

global___GetNewDataSinceRequest = GetNewDataSinceRequest

@typing.final
class GetNewDataSinceResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    KEYS_FIELD_NUMBER: builtins.int
    TIMESTAMPS_FIELD_NUMBER: builtins.int
    LABELS_FIELD_NUMBER: builtins.int
    @property
    def keys(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    @property
    def timestamps(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    @property
    def labels(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        keys: collections.abc.Iterable[builtins.int] | None = ...,
        timestamps: collections.abc.Iterable[builtins.int] | None = ...,
        labels: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["keys", b"keys", "labels", b"labels", "timestamps", b"timestamps"]) -> None: ...

global___GetNewDataSinceResponse = GetNewDataSinceResponse

@typing.final
class GetDataInIntervalRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATASET_ID_FIELD_NUMBER: builtins.int
    START_TIMESTAMP_FIELD_NUMBER: builtins.int
    END_TIMESTAMP_FIELD_NUMBER: builtins.int
    dataset_id: builtins.str
    start_timestamp: builtins.int
    end_timestamp: builtins.int
    def __init__(
        self,
        *,
        dataset_id: builtins.str = ...,
        start_timestamp: builtins.int = ...,
        end_timestamp: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["dataset_id", b"dataset_id", "end_timestamp", b"end_timestamp", "start_timestamp", b"start_timestamp"]) -> None: ...

global___GetDataInIntervalRequest = GetDataInIntervalRequest

@typing.final
class GetDataInIntervalResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    KEYS_FIELD_NUMBER: builtins.int
    TIMESTAMPS_FIELD_NUMBER: builtins.int
    LABELS_FIELD_NUMBER: builtins.int
    @property
    def keys(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    @property
    def timestamps(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    @property
    def labels(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        keys: collections.abc.Iterable[builtins.int] | None = ...,
        timestamps: collections.abc.Iterable[builtins.int] | None = ...,
        labels: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["keys", b"keys", "labels", b"labels", "timestamps", b"timestamps"]) -> None: ...

global___GetDataInIntervalResponse = GetDataInIntervalResponse

@typing.final
class GetDataPerWorkerRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATASET_ID_FIELD_NUMBER: builtins.int
    WORKER_ID_FIELD_NUMBER: builtins.int
    TOTAL_WORKERS_FIELD_NUMBER: builtins.int
    START_TIMESTAMP_FIELD_NUMBER: builtins.int
    END_TIMESTAMP_FIELD_NUMBER: builtins.int
    dataset_id: builtins.str
    worker_id: builtins.int
    total_workers: builtins.int
    start_timestamp: builtins.int
    """value unset or set with default value means no limit
    start_timestamp is inclusive, end_timestamp is exclusive
    """
    end_timestamp: builtins.int
    def __init__(
        self,
        *,
        dataset_id: builtins.str = ...,
        worker_id: builtins.int = ...,
        total_workers: builtins.int = ...,
        start_timestamp: builtins.int = ...,
        end_timestamp: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["dataset_id", b"dataset_id", "end_timestamp", b"end_timestamp", "start_timestamp", b"start_timestamp", "total_workers", b"total_workers", "worker_id", b"worker_id"]) -> None: ...

global___GetDataPerWorkerRequest = GetDataPerWorkerRequest

@typing.final
class GetDataPerWorkerResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    KEYS_FIELD_NUMBER: builtins.int
    @property
    def keys(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        keys: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["keys", b"keys"]) -> None: ...

global___GetDataPerWorkerResponse = GetDataPerWorkerResponse

@typing.final
class GetDatasetSizeRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATASET_ID_FIELD_NUMBER: builtins.int
    START_TIMESTAMP_FIELD_NUMBER: builtins.int
    END_TIMESTAMP_FIELD_NUMBER: builtins.int
    dataset_id: builtins.str
    start_timestamp: builtins.int
    """value unset or set with default value means no limit
    start_timestamp is inclusive, end_timestamp is exclusive
    """
    end_timestamp: builtins.int
    def __init__(
        self,
        *,
        dataset_id: builtins.str = ...,
        start_timestamp: builtins.int = ...,
        end_timestamp: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["dataset_id", b"dataset_id", "end_timestamp", b"end_timestamp", "start_timestamp", b"start_timestamp"]) -> None: ...

global___GetDatasetSizeRequest = GetDatasetSizeRequest

@typing.final
class GetDatasetSizeResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SUCCESS_FIELD_NUMBER: builtins.int
    NUM_KEYS_FIELD_NUMBER: builtins.int
    success: builtins.bool
    num_keys: builtins.int
    def __init__(
        self,
        *,
        success: builtins.bool = ...,
        num_keys: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["num_keys", b"num_keys", "success", b"success"]) -> None: ...

global___GetDatasetSizeResponse = GetDatasetSizeResponse

@typing.final
class DatasetAvailableRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATASET_ID_FIELD_NUMBER: builtins.int
    dataset_id: builtins.str
    def __init__(
        self,
        *,
        dataset_id: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["dataset_id", b"dataset_id"]) -> None: ...

global___DatasetAvailableRequest = DatasetAvailableRequest

@typing.final
class DatasetAvailableResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    AVAILABLE_FIELD_NUMBER: builtins.int
    available: builtins.bool
    def __init__(
        self,
        *,
        available: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["available", b"available"]) -> None: ...

global___DatasetAvailableResponse = DatasetAvailableResponse

@typing.final
class RegisterNewDatasetRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATASET_ID_FIELD_NUMBER: builtins.int
    FILESYSTEM_WRAPPER_TYPE_FIELD_NUMBER: builtins.int
    FILE_WRAPPER_TYPE_FIELD_NUMBER: builtins.int
    DESCRIPTION_FIELD_NUMBER: builtins.int
    BASE_PATH_FIELD_NUMBER: builtins.int
    VERSION_FIELD_NUMBER: builtins.int
    FILE_WRAPPER_CONFIG_FIELD_NUMBER: builtins.int
    IGNORE_LAST_TIMESTAMP_FIELD_NUMBER: builtins.int
    FILE_WATCHER_INTERVAL_FIELD_NUMBER: builtins.int
    dataset_id: builtins.str
    filesystem_wrapper_type: builtins.str
    file_wrapper_type: builtins.str
    description: builtins.str
    base_path: builtins.str
    version: builtins.str
    file_wrapper_config: builtins.str
    ignore_last_timestamp: builtins.bool
    file_watcher_interval: builtins.int
    def __init__(
        self,
        *,
        dataset_id: builtins.str = ...,
        filesystem_wrapper_type: builtins.str = ...,
        file_wrapper_type: builtins.str = ...,
        description: builtins.str = ...,
        base_path: builtins.str = ...,
        version: builtins.str = ...,
        file_wrapper_config: builtins.str = ...,
        ignore_last_timestamp: builtins.bool = ...,
        file_watcher_interval: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["base_path", b"base_path", "dataset_id", b"dataset_id", "description", b"description", "file_watcher_interval", b"file_watcher_interval", "file_wrapper_config", b"file_wrapper_config", "file_wrapper_type", b"file_wrapper_type", "filesystem_wrapper_type", b"filesystem_wrapper_type", "ignore_last_timestamp", b"ignore_last_timestamp", "version", b"version"]) -> None: ...

global___RegisterNewDatasetRequest = RegisterNewDatasetRequest

@typing.final
class RegisterNewDatasetResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SUCCESS_FIELD_NUMBER: builtins.int
    success: builtins.bool
    def __init__(
        self,
        *,
        success: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["success", b"success"]) -> None: ...

global___RegisterNewDatasetResponse = RegisterNewDatasetResponse

@typing.final
class GetCurrentTimestampResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TIMESTAMP_FIELD_NUMBER: builtins.int
    timestamp: builtins.int
    def __init__(
        self,
        *,
        timestamp: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["timestamp", b"timestamp"]) -> None: ...

global___GetCurrentTimestampResponse = GetCurrentTimestampResponse

@typing.final
class DeleteDatasetResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SUCCESS_FIELD_NUMBER: builtins.int
    success: builtins.bool
    def __init__(
        self,
        *,
        success: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["success", b"success"]) -> None: ...

global___DeleteDatasetResponse = DeleteDatasetResponse

@typing.final
class DeleteDataRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATASET_ID_FIELD_NUMBER: builtins.int
    KEYS_FIELD_NUMBER: builtins.int
    dataset_id: builtins.str
    @property
    def keys(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        dataset_id: builtins.str = ...,
        keys: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["dataset_id", b"dataset_id", "keys", b"keys"]) -> None: ...

global___DeleteDataRequest = DeleteDataRequest

@typing.final
class DeleteDataResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SUCCESS_FIELD_NUMBER: builtins.int
    success: builtins.bool
    def __init__(
        self,
        *,
        success: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["success", b"success"]) -> None: ...

global___DeleteDataResponse = DeleteDataResponse
