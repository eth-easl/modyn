from concurrent import futures
from threading import Thread

import grpc

from dynamicdatasets.preprocess.preprocess_pb2 import PreprocessRequest, PreprocessResponse
from dynamicdatasets.preprocess.preprocess_pb2_grpc import PreprocessServicer, add_PreprocessServicer_to_server
from dynamicdatasets.preprocess.preprocessor import Preprocessor


class PreprocessServicer(PreprocessServicer):
    """Provides methods that implement functionality of the preprocess server."""

    def __init__(self, config: dict, preprocess_function: str):
        super().__init__()
        self.__config = config
        self.__preprocessor = Preprocessor(
            config, preprocess_function)

    def Preprocess(self, request: PreprocessRequest, context):
        print("Preprocessing data")
        self.__preprocessor.preprocess(request.value)
        return PreprocessResponse()


def serve(config_dict):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_PreprocessServicer_to_server(
        PreprocessServicer(
            config_dict,
            config_dict['preprocess']['function']),
        server)
    print('Starting server. Listening on port ' +
          config_dict['preprocess']['port'])
    server.add_insecure_port('[::]:' + config_dict['preprocess']['port'])
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    import sys
    import yaml
    import logging

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python preprocess_server.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    serve(config)
