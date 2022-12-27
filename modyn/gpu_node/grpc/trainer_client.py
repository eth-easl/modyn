import grpc
import time

from modyn.gpu_node.grpc.trainer_server_pb2_grpc import TrainerServerStub
from modyn.gpu_node.grpc.trainer_server_pb2 import RegisterTrainServerRequest, Data, TrainerAvailableRequest, VarTypeParameter, CheckpointInfo, StartTrainingRequest

class TrainerClient:

    def __init__(self):
        self._trainer_stub = TrainerServerStub(grpc.insecure_channel("127.0.0.1:1222"))

    def check_trainer_available(self):
        req = TrainerAvailableRequest()
        response = self._trainer_stub.trainer_available(req)
        return response.available


    def register_training(self):
        req = RegisterTrainServerRequest(
            model_id="resnet18",
            batch_size=32,
            torch_optimizer='SGD',
            optimizer_parameters= {
                'lr': VarTypeParameter(float_value=0.1),
                'momentum': VarTypeParameter(float_value=0.001)
            },
            model_configuration={
                'num_classes': VarTypeParameter(int_value=10)
            },
            data_info=Data(
                dataset_id="online",
                num_dataloaders=2
            ),
            checkpoint_info=CheckpointInfo(
                checkpoint_interval = 1,
                checkpoint_path = "results"
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
    training_started = client.start_training(training_id)
    print(training_started)
