import json
from modyn.trainer_server.grpc.trainer_server_pb2 import RegisterTrainServerRequest

STATUS_QUERY_MESSAGE = "get_status"

class TrainingInfo:
    def __init__(self, request: RegisterTrainServerRequest) -> None:

        # TODO(fotstrt): add checks here

        self.training_id = request.training_id
        self.dataset_id = request.data_info.dataset_id,
        self.num_dataloaders = request.data_info.num_dataloaders

        self.optimizer_dict = json.loads(request.optimizer_parameters.value)
        self.model_configuration_dict = json.loads(request.model_configuration.value)
        self.criterion_dict = json.loads(request.criterion_parameters.value)

        self.transform_list = [x for x in request.transform_list]

        self.model_id = request.model_id
        self.torch_optimizer = request.torch_optimizer
        self.batch_size = request.batch_size
        self.torch_criterion = request.torch_criterion

        self.checkpoint_path = request.checkpoint_info.checkpoint_path
        self.checkpoint_interval = request.checkpoint_info.checkpoint_interval


class TrainingProcessInfo:
    def __init__(self, process_handler, exception_queue, status_query_queue, status_response_queue):

        self.process_handler = process_handler
        self.exception_queue = exception_queue
        self.status_query_queue = status_query_queue
        self.status_response_queue = status_response_queue