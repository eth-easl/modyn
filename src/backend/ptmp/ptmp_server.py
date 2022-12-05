import grpc
from concurrent import futures
import os
import sys
from pathlib import Path
import logging

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))

from backend.ptmp.ptmp_pb2_grpc import PostTrainingMetadataProcessorServicer, \
    add_PostTrainingMetadataProcessorServicer_to_server
from backend.ptmp.ptmp_pb2 import PostTrainingMetadataRequest, PostTrainingMetadataResponse
from utils import my_import

logging.basicConfig(format='%(asctime)s %(message)s')


class PostTrainingMetadataProcessor(
        PostTrainingMetadataProcessorServicer):
    """Provides methods that implement functionality of PostTrainingMetadataProcessor server."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.__config = config
        processor_module = my_import('backend.ptmp.processor')
        self.__processor = getattr(
            processor_module,
            config['ptmp']['processor'])(config)

    def ProcessPostTrainingMetadata(
            self, request: PostTrainingMetadataRequest, context):
        self.__processor.process_post_training_metadata(
            request.training_id, request.data)
        return PostTrainingMetadataResponse()


def serve(config: dict) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_PostTrainingMetadataProcessorServicer_to_server(
        PostTrainingMetadataProcessor(config), server)
    logging.info('Starting server. Listening on port .' + config["ptmp"]["port"])
    server.add_insecure_port(f'[::]:{config["ptmp"]["port"]}')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    import sys
    import yaml
    import logging

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python ptmp_server.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    serve(config)
