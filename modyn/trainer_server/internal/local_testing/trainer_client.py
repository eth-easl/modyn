import json
import threading
import time

import grpc
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import (
    CheckpointInfo,
    Data,
    JsonString,
    PythonString,
    StartTrainingRequest,
    TrainerAvailableRequest,
    TrainingStatusRequest,
)
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2_grpc import TrainerServerStub

MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024


class TrainerClient:

    """
    A class to test the grpc server of the gpu node.
    TODO(fotstrt): remove when the supervisor-gpu node communication is fixed
    """

    def __init__(self) -> None:
        self._trainer_stub = TrainerServerStub(
            grpc.insecure_channel(
                "127.0.0.1:5001",
                options=[("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH)],
            )
        )

    def check_trainer_available(self) -> bool:
        req = TrainerAvailableRequest()
        response = self._trainer_stub.trainer_available(req)
        return response.available

    def start_training(self, pipeline_id: int, trigger_id: int) -> bool:
        bytes_parser = """import time\ndef bytes_parser_function(x):\n\treturn x"""

        transforms = [
            "transforms.ToTensor()",
            "transforms.Normalize((0.1307,), (0.3081,))",
        ]

        optimizer_parameters = {"lr": 0.1, "momentum": 0.001}

        model_configuration = {"num_classes": 10}

        req = StartTrainingRequest(
            pipeline_id=pipeline_id,
            trigger_id=trigger_id,
            device="cpu",
            model_id="ResNet18",
            batch_size=32,
            torch_optimizer="SGD",
            torch_criterion="CrossEntropyLoss",
            criterion_parameters=JsonString(value=json.dumps({})),
            optimizer_parameters=JsonString(value=json.dumps(optimizer_parameters)),
            model_configuration=JsonString(value=json.dumps(model_configuration)),
            data_info=Data(dataset_id="MNISTDataset", num_dataloaders=2),
            checkpoint_info=CheckpointInfo(checkpoint_interval=10, checkpoint_path="results"),
            transform_list=transforms,
            bytes_parser=PythonString(value=bytes_parser),
            use_pretrained_model=False
        )

        response = self._trainer_stub.start_training(req)
        return response.training_started, response.training_id

    def get_training_status(self, training_id: int) -> None:
        req = TrainingStatusRequest(training_id=training_id)
        self._trainer_stub.get_training_status(req)


if __name__ == "__main__":
    client = TrainerClient()
    is_available = client.check_trainer_available()
    if is_available:
        success, training_id = client.start_training(1,1)
        print(success, training_id)
        if success:
            time.sleep(10)
            client.get_training_status(training_id)
