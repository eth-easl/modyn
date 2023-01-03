import json
from modyn.gpu_node.grpc.trainer_server_pb2 import RegisterTrainServerRequest


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
