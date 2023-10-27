"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import sys
import typing

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class Empty(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___Empty = Empty

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
class StrategyConfig(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAME_FIELD_NUMBER: builtins.int
    ZIP_FIELD_NUMBER: builtins.int
    ZIP_ALGORITHM_FIELD_NUMBER: builtins.int
    CONFIG_FIELD_NUMBER: builtins.int
    name: builtins.str
    zip: builtins.bool
    zip_algorithm: builtins.str
    @property
    def config(self) -> global___JsonString: ...
    def __init__(
        self,
        *,
        name: builtins.str = ...,
        zip: builtins.bool | None = ...,
        zip_algorithm: builtins.str | None = ...,
        config: global___JsonString | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_config", b"_config", "_zip", b"_zip", "_zip_algorithm", b"_zip_algorithm", "config", b"config", "zip", b"zip", "zip_algorithm", b"zip_algorithm"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_config", b"_config", "_zip", b"_zip", "_zip_algorithm", b"_zip_algorithm", "config", b"config", "name", b"name", "zip", b"zip", "zip_algorithm", b"zip_algorithm"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_config", b"_config"]) -> typing_extensions.Literal["config"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_zip", b"_zip"]) -> typing_extensions.Literal["zip"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_zip_algorithm", b"_zip_algorithm"]) -> typing_extensions.Literal["zip_algorithm"] | None: ...

global___StrategyConfig = StrategyConfig

@typing_extensions.final
class DataInformRequest(google.protobuf.message.Message):
    """// TODO(#302): Remove this when reworking pipeline registration
    message ModelStoragePolicyInfo {
      StrategyConfig full_model_strategy_config = 1;
      optional StrategyConfig incremental_model_strategy_config = 2;
      optional int32 full_model_interval = 3;
    }
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_ID_FIELD_NUMBER: builtins.int
    KEYS_FIELD_NUMBER: builtins.int
    TIMESTAMPS_FIELD_NUMBER: builtins.int
    LABELS_FIELD_NUMBER: builtins.int
    pipeline_id: builtins.int
    @property
    def keys(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    @property
    def timestamps(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    @property
    def labels(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        pipeline_id: builtins.int = ...,
        keys: collections.abc.Iterable[builtins.int] | None = ...,
        timestamps: collections.abc.Iterable[builtins.int] | None = ...,
        labels: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["keys", b"keys", "labels", b"labels", "pipeline_id", b"pipeline_id", "timestamps", b"timestamps"]) -> None: ...

global___DataInformRequest = DataInformRequest

@typing_extensions.final
class DataInformResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    LOG_FIELD_NUMBER: builtins.int
    @property
    def log(self) -> global___JsonString: ...
    def __init__(
        self,
        *,
        log: global___JsonString | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["log", b"log"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["log", b"log"]) -> None: ...

global___DataInformResponse = DataInformResponse

@typing_extensions.final
class TriggerResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRIGGER_ID_FIELD_NUMBER: builtins.int
    LOG_FIELD_NUMBER: builtins.int
    trigger_id: builtins.int
    @property
    def log(self) -> global___JsonString: ...
    def __init__(
        self,
        *,
        trigger_id: builtins.int = ...,
        log: global___JsonString | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["log", b"log"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["log", b"log", "trigger_id", b"trigger_id"]) -> None: ...

global___TriggerResponse = TriggerResponse

@typing_extensions.final
class GetSamplesRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_ID_FIELD_NUMBER: builtins.int
    TRIGGER_ID_FIELD_NUMBER: builtins.int
    PARTITION_ID_FIELD_NUMBER: builtins.int
    WORKER_ID_FIELD_NUMBER: builtins.int
    pipeline_id: builtins.int
    trigger_id: builtins.int
    partition_id: builtins.int
    worker_id: builtins.int
    def __init__(
        self,
        *,
        pipeline_id: builtins.int = ...,
        trigger_id: builtins.int = ...,
        partition_id: builtins.int = ...,
        worker_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["partition_id", b"partition_id", "pipeline_id", b"pipeline_id", "trigger_id", b"trigger_id", "worker_id", b"worker_id"]) -> None: ...

global___GetSamplesRequest = GetSamplesRequest

@typing_extensions.final
class SamplesResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRAINING_SAMPLES_SUBSET_FIELD_NUMBER: builtins.int
    TRAINING_SAMPLES_WEIGHTS_FIELD_NUMBER: builtins.int
    @property
    def training_samples_subset(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    @property
    def training_samples_weights(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]: ...
    def __init__(
        self,
        *,
        training_samples_subset: collections.abc.Iterable[builtins.int] | None = ...,
        training_samples_weights: collections.abc.Iterable[builtins.float] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["training_samples_subset", b"training_samples_subset", "training_samples_weights", b"training_samples_weights"]) -> None: ...

global___SamplesResponse = SamplesResponse

@typing_extensions.final
class GetNumberOfSamplesRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_ID_FIELD_NUMBER: builtins.int
    TRIGGER_ID_FIELD_NUMBER: builtins.int
    pipeline_id: builtins.int
    trigger_id: builtins.int
    def __init__(
        self,
        *,
        pipeline_id: builtins.int = ...,
        trigger_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["pipeline_id", b"pipeline_id", "trigger_id", b"trigger_id"]) -> None: ...

global___GetNumberOfSamplesRequest = GetNumberOfSamplesRequest

@typing_extensions.final
class NumberOfSamplesResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NUM_SAMPLES_FIELD_NUMBER: builtins.int
    num_samples: builtins.int
    def __init__(
        self,
        *,
        num_samples: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["num_samples", b"num_samples"]) -> None: ...

global___NumberOfSamplesResponse = NumberOfSamplesResponse

@typing_extensions.final
class GetStatusBarScaleRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_ID_FIELD_NUMBER: builtins.int
    pipeline_id: builtins.int
    def __init__(
        self,
        *,
        pipeline_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["pipeline_id", b"pipeline_id"]) -> None: ...

global___GetStatusBarScaleRequest = GetStatusBarScaleRequest

@typing_extensions.final
class StatusBarScaleResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STATUS_BAR_SCALE_FIELD_NUMBER: builtins.int
    status_bar_scale: builtins.int
    def __init__(
        self,
        *,
        status_bar_scale: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["status_bar_scale", b"status_bar_scale"]) -> None: ...

global___StatusBarScaleResponse = StatusBarScaleResponse

@typing_extensions.final
class GetNumberOfPartitionsRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_ID_FIELD_NUMBER: builtins.int
    TRIGGER_ID_FIELD_NUMBER: builtins.int
    pipeline_id: builtins.int
    trigger_id: builtins.int
    def __init__(
        self,
        *,
        pipeline_id: builtins.int = ...,
        trigger_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["pipeline_id", b"pipeline_id", "trigger_id", b"trigger_id"]) -> None: ...

global___GetNumberOfPartitionsRequest = GetNumberOfPartitionsRequest

@typing_extensions.final
class NumberOfPartitionsResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NUM_PARTITIONS_FIELD_NUMBER: builtins.int
    num_partitions: builtins.int
    def __init__(
        self,
        *,
        num_partitions: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["num_partitions", b"num_partitions"]) -> None: ...

global___NumberOfPartitionsResponse = NumberOfPartitionsResponse

@typing_extensions.final
class GetAvailableLabelsRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_ID_FIELD_NUMBER: builtins.int
    pipeline_id: builtins.int
    def __init__(
        self,
        *,
        pipeline_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["pipeline_id", b"pipeline_id"]) -> None: ...

global___GetAvailableLabelsRequest = GetAvailableLabelsRequest

@typing_extensions.final
class AvailableLabelsResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    AVAILABLE_LABELS_FIELD_NUMBER: builtins.int
    @property
    def available_labels(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        available_labels: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["available_labels", b"available_labels"]) -> None: ...

global___AvailableLabelsResponse = AvailableLabelsResponse

@typing_extensions.final
class GetSelectionStrategyRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_ID_FIELD_NUMBER: builtins.int
    pipeline_id: builtins.int
    def __init__(
        self,
        *,
        pipeline_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["pipeline_id", b"pipeline_id"]) -> None: ...

global___GetSelectionStrategyRequest = GetSelectionStrategyRequest

@typing_extensions.final
class SelectionStrategyResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DOWNSAMPLING_ENABLED_FIELD_NUMBER: builtins.int
    STRATEGY_NAME_FIELD_NUMBER: builtins.int
    DOWNSAMPLER_CONFIG_FIELD_NUMBER: builtins.int
    downsampling_enabled: builtins.bool
    strategy_name: builtins.str
    @property
    def downsampler_config(self) -> global___JsonString: ...
    def __init__(
        self,
        *,
        downsampling_enabled: builtins.bool = ...,
        strategy_name: builtins.str = ...,
        downsampler_config: global___JsonString | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["downsampler_config", b"downsampler_config"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["downsampler_config", b"downsampler_config", "downsampling_enabled", b"downsampling_enabled", "strategy_name", b"strategy_name"]) -> None: ...

global___SelectionStrategyResponse = SelectionStrategyResponse

@typing_extensions.final
class UsesWeightsRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PIPELINE_ID_FIELD_NUMBER: builtins.int
    pipeline_id: builtins.int
    def __init__(
        self,
        *,
        pipeline_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["pipeline_id", b"pipeline_id"]) -> None: ...

global___UsesWeightsRequest = UsesWeightsRequest

@typing_extensions.final
class UsesWeightsResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    USES_WEIGHTS_FIELD_NUMBER: builtins.int
    uses_weights: builtins.bool
    def __init__(
        self,
        *,
        uses_weights: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["uses_weights", b"uses_weights"]) -> None: ...

global___UsesWeightsResponse = UsesWeightsResponse

@typing_extensions.final
class SeedSelectorRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SEED_FIELD_NUMBER: builtins.int
    seed: builtins.int
    def __init__(
        self,
        *,
        seed: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["seed", b"seed"]) -> None: ...

global___SeedSelectorRequest = SeedSelectorRequest

@typing_extensions.final
class SeedSelectorResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SUCCESS_FIELD_NUMBER: builtins.int
    success: builtins.bool
    def __init__(
        self,
        *,
        success: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["success", b"success"]) -> None: ...

global___SeedSelectorResponse = SeedSelectorResponse
