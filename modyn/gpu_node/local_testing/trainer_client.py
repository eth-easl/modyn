import json
import grpc

from modyn.gpu_node.grpc.trainer_server_pb2_grpc import TrainerServerStub
from modyn.gpu_node.grpc.trainer_server_pb2 import (
    JsonString,
    RegisterTrainServerRequest,
    Data,
    TorchvisionTransform,
    TrainerAvailableRequest,
    CheckpointInfo,
    StartTrainingRequest
)


class TrainerClient:

    """
    A class to test the grpc server of the gpu node.
    """

    def __init__(self):
        self._trainer_stub = TrainerServerStub(grpc.insecure_channel("127.0.0.1:5001"))

    def check_trainer_available(self):
        req = TrainerAvailableRequest()
        response = self._trainer_stub.trainer_available(req)
        return response.available

    def register_training(self, training_id):

        transforms = [
            TorchvisionTransform(function="ToTensor", args=JsonString(value=json.dumps({}))),
            TorchvisionTransform(function="Normalize", args=JsonString(value=json.dumps({"mean": (0.1307,), "std": (0.3081,)}))),
        ]

        optimizer_parameters = {
            'lr': 0.1,
            'momentum': 0.001
        }

        model_configuration = {
            'num_classes': 10
        }

        req = RegisterTrainServerRequest(
            training_id=training_id,
            model_id="ResNet18",
            batch_size=32,
            torch_optimizer='SGD',
            optimizer_parameters=JsonString(value=json.dumps(optimizer_parameters)),
            model_configuration=JsonString(value=json.dumps(model_configuration)),
            data_info=Data(
                dataset_id="MNISTDataset",
                num_dataloaders=2
            ),
            checkpoint_info=CheckpointInfo(
                checkpoint_interval=10,
                checkpoint_path="results"
            ),
            transform_list = transforms
        )

        response = self._trainer_stub.register(req)
        return response.success

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
    training_id = 10
    if is_available:
        success = client.register_training(training_id)
        print(success)
        if success:
            training_started = client.start_training(training_id)
            print(training_started)

            while (not client.check_trainer_available()):
                pass

            training_started = client.start_training(training_id)
            print(training_started)

            print("started again!")
