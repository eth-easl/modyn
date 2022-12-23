import grpc
import time

from modyn.gpu_node.grpc.trainer_server_pb2_grpc import TrainerServerStub
from modyn.gpu_node.grpc.trainer_server_pb2 import TrainerServerRequest, TrainingHyperparameters, Data, TrainerAvailableRequest

class TrainerClient:

    def __init__(self):
        self._trainer_stub = TrainerServerStub(grpc.insecure_channel("127.0.0.1:1222"))

    def check_trainer_available(self):
        req = TrainerAvailableRequest()
        response = self._trainer_stub.trainer_available(req)
        return response.available

    def start_training(self):
        req = TrainerServerRequest(
            model_id="ResNet18",
            hyperparameters=TrainingHyperparameters(
                batch_size=32,
                learning_rate=0.1
            ),
            checkpoint_path="",
            model_configuration={},
            data_info=Data(
                dataset_id="online",
                num_dataloaders=2
            )
        )
        response = self._trainer_stub.start_training(req)

        return response.training_id

if __name__ == "__main__":
    client = TrainerClient()
    is_available = client.check_trainer_available()
    if is_available:
        training_id = client.start_training()
    print(training_id)
