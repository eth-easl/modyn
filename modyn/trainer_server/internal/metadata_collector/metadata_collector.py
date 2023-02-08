from typing import Any, Callable

from modyn.trainer_server.internal.mocks.mock_metadata_processor import (
    MockMetadataProcessorServer,
    PerSampleMetadata,
    PerTriggerMetadata,
    TrainingMetadataRequest,
)
from modyn.trainer_server.internal.utils.metric_type import MetricType


class MetadataCollector:
    def __init__(self, pipeline_id: str, trigger_id: int):
        self._per_sample_metadata_dict: dict[MetricType, Any] = {}
        self._per_trigger_metadata: dict[MetricType, Any] = {}
        self._metric_handlers: dict[MetricType, Callable] = {}
        self._pipeline_id = pipeline_id
        self._trigger_id = trigger_id

        self.register_metric_handlers()
        # TODO(#139): remove this when the MetadataProcessor is fixed
        self._metadata_processor_stub = MockMetadataProcessorServer()

    def register_metric_handlers(self) -> None:
        # can register new handlers here
        self._metric_handlers[MetricType.LOSS] = self.send_loss

    def add_per_sample_metadata_for_batch(self, metric: MetricType, sample_ids: list[str], metadata: list[Any]) -> None:
        # We expect a list of the sample ids for a batch, along with a list of their metadata.
        # Also, we expect metadata are already in a proper format for grpc messages
        # (no need for extra serialization, or type transform)
        assert len(sample_ids) == len(metadata), "Sample id and per-sample metadata lists do not match"

        if metric not in self._per_sample_metadata_dict:
            self._per_sample_metadata_dict[metric] = {}

        for sample_id, metric_value in zip(sample_ids, metadata):
            self._per_sample_metadata_dict[metric][sample_id] = metric_value

    def add_per_trigger_metadata(self, metric: MetricType, metadata: Any) -> None:
        self._per_trigger_metadata[metric] = metadata

    def send_metadata(self, metric: MetricType) -> None:
        self._metric_handlers[metric]()

    def send_loss(self) -> None:
        # In the future, when we add more metadata that follows the same pattern, 
        # we might want to extract out the basic functionality 
        # of sending a `TrainingMetadataRequest` with a respective kwarg.
        
        metric_dict = self._per_sample_metadata_dict[MetricType.LOSS]
        per_trigger_metric = self._per_trigger_metadata[MetricType.LOSS]

        per_sample_metadata_list = [
            PerSampleMetadata(sample_id=sample_id, loss=loss) for sample_id, loss in metric_dict.items()
        ]

        # TODO(#139): replace this with grpc calls to the MetadataProcessor
        metadata_request = TrainingMetadataRequest(
            pipeline_id=self._pipeline_id,
            trigger_id=self._trigger_id,
            per_sample_metadata=per_sample_metadata_list,
            per_trigger_metadata=PerTriggerMetadata(loss=per_trigger_metric),
        )

        self._metadata_processor_stub.send_metadata(metadata_request)

    def cleanup(self) -> None:
        self._per_sample_metadata_dict.clear()
        self._per_trigger_metadata.clear()
