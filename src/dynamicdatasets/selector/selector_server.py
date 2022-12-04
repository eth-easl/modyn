from selector_pb2_grpc import SelectorServicer, add_SelectorServicer_to_server
from selector_pb2 import SamplesResponse, TrainingResponse
import grpc
from new_data_selector import NewDataSelector
from concurrent import futures


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
        print("Registering training with request - " + str(request))
        training_id = self._selector.register_training(
            request.training_set_size, request.num_workers)
        return TrainingResponse(training_id=training_id)

    # def get_sample_keys(self, training_id: int, training_set_number: int,
    # worker_id: int) -> list():
    def get_sample_keys(self, request, context):
        print("Fetching samples for request - " + str(request))
        samples_keys = self._selector.get_sample_keys(
            request.training_id, request.training_set_number, request.worker_id)
        return SamplesResponse(training_samples_subset=samples_keys)


def serve(config: dict):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_SelectorServicer_to_server(
        SelectorServicer(config), server)
    print(
        'Starting server. Listening on port .' +
        config['selector']['port'])
    server.add_insecure_port('[::]:' + config['selector']['port'])
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    import sys
    import yaml
    import logging

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python selector_server.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    serve(config)
