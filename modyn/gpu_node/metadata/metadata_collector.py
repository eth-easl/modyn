import json
import grpc
from typing import Any
from modyn.backend.ptmp.ptmp_pb2 import PostTrainingMetadataRequest

from modyn.backend.ptmp.ptmp_pb2_grpc import PostTrainingMetadataProcessorStub

class MetadataCollector:

    def __init__(self, addr: str, training_id: int):

        self._metadata_dict = {}
        self._training_id = training_id
        self._processor_stub = PostTrainingMetadataProcessorStub(grpc.insecure_channel(addr))

    def add_metadata(self, key: str, value: Any):

        self._metadata_dict[key] = value

    def send_metadata(self):

        req = PostTrainingMetadataRequest(
            training_id=self._training_id,
            data=json.dumps(self._metadata_dict)
        )
        self._processor_stub.ProcessPostTrainingMetadata(req)
