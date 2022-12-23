import grpc
import time

from modyn.gpu_node.grpc.trainer_server_pb2_grpc import TrainerServerStub
from modyn.gpu_node.grpc.trainer_server_pb2 import TrainerServerRequest, TrainingHyperparameters, Data


def start_training():
    trainer_stub = TrainerServerStub(grpc.insecure_channel("127.0.0.1:1222"))
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
    response = trainer_stub.start_training(req)

    # time.sleep(20)

    # response = trainer_stub.start_training(req)

    return response.training_id

if __name__ == "__main__":
    training_id = start_training()
    print(training_id)
