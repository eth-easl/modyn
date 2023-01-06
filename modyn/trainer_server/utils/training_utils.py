# pylint: disable=no-name-in-module, too-many-instance-attributes
import json
import multiprocessing as mp

from modyn.trainer_server.grpc.generated.trainer_server_pb2 import RegisterTrainServerRequest
from modyn.utils.utils import dynamic_module_import

STATUS_QUERY_MESSAGE = "get_status"


class TrainingInfo:
    def __init__(self, request: RegisterTrainServerRequest) -> None:

        self.training_id = request.training_id
        self.dataset_id = request.data_info.dataset_id
        self.num_dataloaders = request.data_info.num_dataloaders

        self.optimizer_dict = json.loads(request.optimizer_parameters.value)
        self.model_configuration_dict = json.loads(request.model_configuration.value)
        self.criterion_dict = json.loads(request.criterion_parameters.value)

        self.transform_list = list(request.transform_list)

        self.model_id = request.model_id
        model_module = dynamic_module_import("modyn.models")
        if not hasattr(model_module, self.model_id):
            raise ValueError(f"Model {self.model_id} not available!")

        self.torch_optimizer = request.torch_optimizer
        self.batch_size = request.batch_size
        self.torch_criterion = request.torch_criterion

        self.checkpoint_path = request.checkpoint_info.checkpoint_path
        self.checkpoint_interval = request.checkpoint_info.checkpoint_interval


class TrainingProcessInfo:
    def __init__(
        self,
        process_handler: mp.Process,
        exception_queue: mp.Queue,
        status_query_queue: mp.Queue,
        status_response_queue: mp.Queue
    ):

        self.process_handler = process_handler
        self.exception_queue = exception_queue
        self.status_query_queue = status_query_queue
        self.status_response_queue = status_response_queue
