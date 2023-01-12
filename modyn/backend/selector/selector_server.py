import logging
import os
import sys
from pathlib import Path

from modyn.backend.selector.selector_strategy import SelectorStrategy

from modyn.backend.selector.internal.grpc.generated.selector_pb2_grpc import SelectorServicer  # noqa: E402, E501
# Pylint cannot handle the auto-generated gRPC files, apparently.
# pylint: disable-next=no-name-in-module
from modyn.backend.selector.internal.grpc.generated.selector_pb2 import RegisterTrainingRequest, GetSamplesRequest, SamplesResponse, TrainingResponse  # noqa: E402, E501

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))


logging.basicConfig(format='%(asctime)s %(message)s')


class SelectorGRPCServer(SelectorServicer):
    """Provides methods that implement functionality of the metadata server."""

    def __init__(self, strategy: SelectorStrategy):
        # selector_module = dynamic_module_import(
        #     f"modyn.backend.selector.custom_selectors.{config['selector']['package']}")
        # self._selector = getattr(selector_module, config['selector']['class'])(config)
        self.selector_strategy = strategy

    def register_training(self, request: RegisterTrainingRequest, context: grpc.ServicerContext) -> TrainingResponse:
        logging.info(f"Registering training with request - {str(request)}")
        training_id = self.selector_strategy.register_training(
            request.training_set_size, request.num_workers)
        return TrainingResponse(training_id=training_id)

    def get_sample_keys(self, request: GetSamplesRequest, context: grpc.ServicerContext) -> SamplesResponse:
        logging.info(f"Fetching samples for request - {str(request)}")
        samples_keys = self.selector_strategy.get_sample_keys(
            request.training_id, request.training_set_number, request.worker_id)
        samples_keys = [sample[0] for sample in samples_keys]
        return SamplesResponse(training_samples_subset=samples_keys)


# def main() -> None:
#     logging.basicConfig(level=logging.NOTSET,
#                         format='[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s',
#                         datefmt='%Y-%m-%d:%H:%M:%S')
#     logger = logging.getLogger(__name__)
#     if len(sys.argv) != 2:
#         logger.error("Usage: python selector_server.py <config_file>")
#         sys.exit(1)

#     with open(sys.argv[1], "r", encoding="utf-8") as file:
#         config = yaml.safe_load(file)

#     serve(config, SelectorGRPCServer(config))


# if __name__ == '__main__':
#     main()
