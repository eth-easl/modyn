import io
from typing import Any
import grpc
import torch

from modyn.trainer_server.internal.mocks.mock_metadata_processor import MockMetadataProcessorServer, PerSampleMetadata, TrainingMetadataRequest

class MetadataCollector:

    def __init__(self, pipeline_id: str, trigger_id: int):

        self._per_sample_metadata_dict = {}
        self._per_trigger_metadata = None
        self._pipeline_id = pipeline_id
        self._trigger_id = trigger_id

        # TODO(): remove this when the MetadataProcessor is fixed
        self._metadata_processor_stub = MockMetadataProcessorServer()

    def add_per_sample_metadata_for_batch(self, sample_id_list: list[str], metadata_list: list[Any]):

        # We expect a list of the sample ids for a batch, along with a list of their metadata.
        # Also, we expect metadata are already in a proper format for grpc messages (no need for extra serialization)
        assert len(sample_id_list) == len(metadata_list)
        for sample_id, metadata in zip(sample_id_list, metadata_list):
            self._per_sample_metadata_dict[sample_id] = metadata

    def add_per_trigger_metadata(self, metadata: Any):
        self._per_trigger_metadata = metadata

    def send_metadata(self):

        per_sample_metadata_list = []
        for sample_id, metadata in  self._per_sample_metadata_dict.items():
            per_sample_metadata_list.append(PerSampleMetadata(sample_id=sample_id, metadata=metadata))

        per_trigger_metadata_serialized = self._serialize_metadata(self._per_trigger_metadata)

        # TODO(): replace this with grpc calls to the MetadataProcessor
        metadata_request = TrainingMetadataRequest(
            pipeline_id=self._pipeline_id,
            trigger_id=self._trigger_id,
            per_sample_metadata=per_sample_metadata_list,
            per_tigger_metadata=per_trigger_metadata_serialized
        )
        self._metadata_processor_stub.send_metadata(metadata_request)
        self._per_sample_metadata_dict.clear()