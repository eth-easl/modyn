from concurrent import futures

import grpc

from dynamicdatasets.metadata.metadata_pb2 import AddMetadataResponse, GetNextResponse
from dynamicdatasets.metadata.metadata_pb2_grpc import MetadataServicer, add_MetadataServicer_to_server


class MetadataServicer(MetadataServicer):
    """Provides methods that implement functionality of the metadata server."""

    def __init__(self, config: dict):
        scorer_module = self.my_import('dynamicdatasets.metadata.scorer')
        self._scorer = getattr(
            scorer_module,
            config['metadata']['scorer'])(config)
        selector_module = self.my_import('dynamicdatasets.metadata.selector')
        self._selector = getattr(
            selector_module,
            config['metadata']['selector'])(config)

    def my_import(self, name):
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    def AddMetadata(self, request, context):
        print("Adding metadata")
        metadata_id = self._scorer.add_training_set(
            request.filename, request.rows)
        if metadata_id is None:
            return AddMetadataResponse(metadataId=-1)
        else:
            return AddMetadataResponse(metadataId=metadata_id)

    def GetNext(self, request, context):
        print("Getting metadata")
        next = self._selector.get_next_training_set()
        if next is None:
            return GetNextResponse(dataMap={})
        else:
            return GetNextResponse(dataMap=next)


def serve(config_dict):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_MetadataServicer_to_server(
        MetadataServicer(config_dict), server)
    print('Starting server. Listening on port 50051.')
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
