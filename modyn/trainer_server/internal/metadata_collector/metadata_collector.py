from typing import Any

from modyn.trainer_server.internal.mocks.mock_metadata_processor import (
    MockMetadataProcessorServer,
    PerSampleLoss,
    TrainingMetadataRequest,
)


class MetadataCollector:
    def __init__(self, pipeline_id: str, trigger_id: int):

        self._per_sample_metadata_dict: dict[str, Any] = {}
        self._per_trigger_metadata: dict[str, Any] = {}
        self._pipeline_id = pipeline_id
        self._trigger_id = trigger_id

        # TODO(): remove this when the MetadataProcessor is fixed
        self._metadata_processor_stub = MockMetadataProcessorServer()

    def add_per_sample_metadata_for_batch(
        self, metric: str, sample_id_list: list[str], metadata_list: list[Any]
    ) -> None:

        # We expect a list of the sample ids for a batch, along with a list of their metadata.
        # Also, we expect metadata are already in a proper format for grpc messages
        # (no need for extra serialization, or type transform)
        assert len(sample_id_list) == len(metadata_list), "Sample id and per-sample metadata lists do not match"

        if metric not in self._per_sample_metadata_dict:
            self._per_sample_metadata_dict[metric] = {}

        for sample_id, metadata in zip(sample_id_list, metadata_list):
            self._per_sample_metadata_dict[metric][sample_id] = metadata

    def add_per_trigger_metadata(self, metric: str, metadata: Any) -> None:
        self._per_trigger_metadata[metric] = metadata

    def send_metadata(self) -> None:

        for metric in self._per_sample_metadata_dict:

            per_sample_metadata_list = []

            if metric == "loss":
                for sample_id, loss in self._per_sample_metadata_dict.items():
                    per_sample_metadata_list.append(PerSampleLoss(sample_id=sample_id, loss=loss))

            # TODO(): replace this with grpc calls to the MetadataProcessor
            metadata_request = TrainingMetadataRequest(
                pipeline_id=self._pipeline_id,
                trigger_id=self._trigger_id,
                per_sample_metadata=per_sample_metadata_list,
            )
            self._metadata_processor_stub.send_metadata(metadata_request)

        for metric, metadata in self._per_trigger_metadata.items():

            metadata_request = TrainingMetadataRequest(
                pipeline_id=self._pipeline_id, trigger_id=self._trigger_id, per_tigger_metadata=metadata
            )

            self._metadata_processor_stub.send_metadata(metadata_request)

        self._per_sample_metadata_dict.clear()
        self._per_trigger_metadata.clear()
