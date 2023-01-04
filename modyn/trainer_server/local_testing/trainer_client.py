import json
import grpc
import time

from modyn.trainer_server.grpc.trainer_server_pb2_grpc import TrainerServerStub
from modyn.trainer_server.grpc.trainer_server_pb2 import (
    IsRunningRequest,
    JsonString,
    RegisterTrainServerRequest,
    Data,
    TrainerAvailableRequest,
    CheckpointInfo,
    StartTrainingRequest,
    TrainingStatusRequest
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

        transforms = ["transforms.ToTensor()",  "transforms.Normalize((0.1307,), (0.3081,))"]

        optimizer_parameters = {
            'lr': 0.1,
            'momentum': 0.001
        }

        model_configuration = {
            'num_classes': 10,
        }

        req = RegisterTrainServerRequest(
            training_id=training_id,
            model_id="ResNet18",
            batch_size=32,
            torch_optimizer='SGD',
            torch_criterion='CrossEntropyLoss',
            criterion_parameters=JsonString(value=json.dumps({})),
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
            transform_list=transforms
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

    def is_running(self, training_id):

        req = IsRunningRequest(training_id=training_id)
        response = self._trainer_stub.is_running(req)
        return response.is_running

    def get_training_status(self, training_id):

        req = TrainingStatusRequest(training_id=training_id)
        response = self._trainer_stub.get_training_status(req)
        print("------------ Response: ")
        print(response.is_running)
        print(response.exception)
        print(response.iteration)
        print(response.state)


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
            time.sleep(10)
            client.get_training_status(training_id)

            # while (client.is_running(training_id)):
            #     print("is_running")

            # while (not client.check_trainer_available()):
            #     pass

            # training_started = client.start_training(training_id)
            # print(training_started)

            # print("started again!")
