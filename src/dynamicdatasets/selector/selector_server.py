from concurrent import futures
from new_data_selector import NewDataSelector

import grpc

from selector_pb2 import SamplesResponse, TrainingResponse
from selector_pb2_grpc import SelectorServicer, add_SelectorServicer_to_server


class SelectorServicer(SelectorServicer):
    """Provides methods that implement functionality of the metadata server."""

    def __init__(self, config: dict):
        # selector_module = self.my_import('dynamicdatasets.selector')
        # self._selector = getattr(selector_module,config['metadata']['selector'])(config)
        self._selector = NewDataSelector(config)

    def my_import(self, name):
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    def register_training(self, request, context):
        print("Registerining training")
        training_id = self._selector.register_training(request.training_set_size, request.num_workers)
        return TrainingResponse(training_id = training_id)

    #def get_sample_keys(self, training_id: int, training_set_number: int, worker_id: int) -> list():
    def get_sample_keys(self, request, context):
        print("Returning samples")
        samples_keys = self._selector.get_sample_keys(request.training_id, request.training_set_number, request.worker_id)
        return SamplesResponse(training_samples_subset = samples_keys)


def serve(config_dict):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_SelectorServicer_to_server(
        SelectorServicer(config_dict), server)
    print('Starting server. Listening on port 5444.')
    server.add_insecure_port('[::]:5444')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    import logging
    import sys
    import yaml

    logging.basicConfig()
    #with open(sys.argv[1], 'r') as stream:
    #    config = yaml.safe_load(stream)
    serve({})