import json
import grpc

from modyn.gpu_node.grpc.trainer_server_pb2_grpc import TrainerServerStub
from modyn.gpu_node.grpc.trainer_server_pb2 import (
    RegisterTrainServerRequest,
    Data,
    TrainerAvailableRequest,
    CheckpointInfo,
    StartTrainingRequest
)


class TrainerClient:

    """
    A class to test the grpc server of the gpu node.
    """

    def __init__(self):
        self._trainer_stub = TrainerServerStub(grpc.insecure_channel("127.0.0.1:1222"))

    def check_trainer_available(self):
        req = TrainerAvailableRequest()
        response = self._trainer_stub.trainer_available(req)
        return response.available

    def register_training(self):

        optimizer_parameters = {
            'lr': 0.1,
            'momentum': 0.001
        }

        model_configuration = {
            'num_classes': 10
        }

        req = RegisterTrainServerRequest(
            model_id="ResNet18",
            batch_size=32,
            torch_optimizer='SGD',
            optimizer_parameters=json.dumps(optimizer_parameters),
            model_configuration=json.dumps(model_configuration),
            data_info=Data(
                dataset_id="MNISTDataset",
                num_dataloaders=2
            ),
            checkpoint_info=CheckpointInfo(
                checkpoint_interval=10,
                checkpoint_path="results"
            )
        )

        response = self._trainer_stub.register(req)
        return response.training_id

    def start_training(self, training_id):

        req = StartTrainingRequest(
            training_id=training_id,
            load_checkpoint_path="results/model_0.pt"
        )
        response = self._trainer_stub.start_training(req)

        return response.training_started


if __name__ == "__main__":
    client = TrainerClient()
    is_available = client.check_trainer_available()
    if is_available:
        training_id = client.register_training()
    print(training_id)
    training_started = client.start_training(training_id)
    print(training_started)

    while (not client.start_training(training_id)):
        pass

    print("started again!")
