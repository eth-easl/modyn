from concurrent import futures
import yaml
import argparse

import grpc
from dynamicdatasets.metadata.metadata_pb2 import BatchData, BatchId
from dynamicdatasets.metadata.metadata_pb2_grpc import MetadataServicer, add_MetadataServicer_to_server


def parse_args():
    parser = argparse.ArgumentParser(description="Feeder")
    parser.add_argument("config", help="Config File")
    args = parser.parse_args()
    return args


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

    def AddBatch(self, request, context):
        print("Adding batch")
        batch_id = self._scorer.add_batch(request.filename, request.rows)
        if batch_id is None:
            return BatchId(batchId=-1)
        else:
            return BatchId(batchId=batch_id)

    def GetBatch(self, request, context):
        print("Getting batch")
        batch = self._selector.get_next_batch()
        if batch is None:
            return BatchData(dataMap={})
        else:
            return BatchData(dataMap=batch)


def serve():
    args = parse_args()
    config = args.config

    with open(config, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_MetadataServicer_to_server(
        MetadataServicer(parsed_yaml), server)
    print('Starting server. Listening on port 50051.')
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
