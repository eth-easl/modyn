import json
import logging
import sys
from time import sleep
from typing import Any

import grpc

# pylint: disable-next=no-name-in-module
from modyn.storage.internal.grpc.generated.storage_pb2 import (
    DatasetAvailableRequest,
    GetCurrentTimestampResponse,
    GetDataInIntervalRequest,
    GetDataInIntervalResponse,
    GetNewDataSinceRequest,
    GetNewDataSinceResponse,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
# pylint: disable-next=no-name-in-module
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import (
    CheckpointInfo,
    Data,
    JsonString,
    PythonString,
    RegisterTrainServerRequest,
    RegisterTrainServerResponse,
    StartTrainingRequest,
    StartTrainingResponse,
    TrainerAvailableRequest,
    TrainingStatusRequest,
    TrainingStatusResponse,
)
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2_grpc import TrainerServerStub
from modyn.utils import grpc_connection_established

logger = logging.getLogger(__name__)


class GRPCHandler:
    def __init__(self, modyn_config: dict):
        self.config = modyn_config
        self.connected_to_storage = False
        self.connected_to_trainer_server = False

        self.init_storage()
        self.init_trainer_server()

    def init_storage(self) -> None:
        assert self.config is not None
        storage_address = f"{self.config['storage']['hostname']}:{self.config['storage']['port']}"
        self.storage_channel = grpc.insecure_channel(storage_address)

        if not grpc_connection_established(self.storage_channel):
            raise ConnectionError(f"Could not establish gRPC connection to storage at {storage_address}.")

        self.storage = StorageStub(self.storage_channel)
        logger.info("Successfully connected to storage.")
        self.connected_to_storage = True

    def init_trainer_server(self) -> None:
        assert self.config is not None
        trainer_server_address = f"{self.config['trainer_server']['hostname']}:{self.config['trainer_server']['port']}"
        self.trainer_server_channel = grpc.insecure_channel(trainer_server_address)

        if not grpc_connection_established(self.trainer_server_channel):
            raise ConnectionError(f"Could not establish gRPC connection to trainer server at {trainer_server_address}.")

        self.trainer_server = TrainerServerStub(self.trainer_server_channel)
        logger.info("Successfully connected to trainer server.")
        self.connected_to_trainer_server = True

    def dataset_available(self, dataset_id: str) -> bool:
        assert self.connected_to_storage, "Tried to check for dataset availability, but no storage connection."
        logger.info(f"Checking whether dataset {dataset_id} is available.")

        response = self.storage.CheckAvailability(DatasetAvailableRequest(dataset_id=dataset_id))

        return response.available

    def get_new_data_since(self, dataset_id: str, timestamp: int) -> list[tuple[str, int]]:
        if not self.connected_to_storage:
            raise ConnectionError("Tried to fetch data from storage, but no connection was made.")

        request = GetNewDataSinceRequest(dataset_id=dataset_id, timestamp=timestamp)
        response: GetNewDataSinceResponse = self.storage.GetNewDataSince(request)

        return zip(response.keys, response.timestamps)

    def get_data_in_interval(self, dataset_id: str, start_timestamp: int, end_timestamp: int) -> list[tuple[str, int]]:
        if not self.connected_to_storage:
            raise ConnectionError("Tried to fetch data from storage, but no connection was made.")

        request = GetDataInIntervalRequest(
            dataset_id=dataset_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )
        response: GetDataInIntervalResponse = self.storage.GetDataInInterval(request)

        return zip(response.keys, response.timestamps)

    def get_time_at_storage(self) -> int:
        if not self.connected_to_storage:
            raise ConnectionError("Tried to fetch data from storage, but no connection was made.")

        response: GetCurrentTimestampResponse = self.storage.GetCurrentTimestamp()

        return response.timestamp

    # pylint: disable-next=unused-argument
    def register_pipeline_at_selector(self, pipeline_config: dict) -> int:
        # TODO(#64): Implement gRPC call.
        return 42

    # pylint: disable-next=unused-argument
    def unregister_pipeline_at_selector(self, pipeline_id: int) -> None:
        #  # TODO(#64,#130): Implement.
        pass

    # pylint: disable-next=unused-argument
    def inform_selector(self, pipeline_id: int, data: list[tuple[str, int]]) -> None:
        # TODO(#64): Implement gRPC call.
        pass

    # pylint: disable-next=unused-argument
    def inform_selector_and_trigger(self, pipeline_id: int, data: list[tuple[str, int]]) -> int:
        # TODO(#64): Implement gRPC call.
        return 42

    def trainer_server_available(self) -> bool:
        if not self.connected_to_trainer_server:
            raise ConnectionError("Tried to check whether server is available, but Supervisor is not even connected!")

        logger.info("Checking whether trainer server is available.")

        request = TrainerAvailableRequest()
        response = self.trainer_server.trainer_available(request)

        logger.info(f"Trainer Server Availability = {response.available}")

        return response.available

    # pylint: disable-next=unused-argument
    def stop_training_at_trainer_server(self, training_id: int) -> None:
        # TODO(#130): Implement this at trainer server.
        logger.error("The trainer server currently does not support remotely stopping training, ignoring.")
        pass

    def register_pipeline_at_trainer_server(self, pipeline_id: int, pipeline_config: dict) -> None:
        if "model_config" in pipeline_config["model"]:
            model_config = json.dumps(pipeline_config["model"]["model_config"])
        else:
            model_config = "{}"

        # TODO(MaxiBoether): Support Optimizer/Criterion in pipeline config
        optimizer_parameters = json.dumps({"lr": 0.1, "momentum": 0.001})

        if "transformations" in pipeline_config["data"]:
            transform_list=pipeline_config["data"]["transformations"]
        else:
            transform_list=[]

        if pipeline_config["checkpointing"]["activated"]:
            if "interval" not in pipeline_config["checkpointing"] or "path" not in pipeline_config["checkpointing"]:
                raise ValueError("Checkpointing is enabled, but interval or path not given.")

            checkpoint_info = CheckpointInfo(checkpoint_interval=pipeline_config["checkpointing"]["interval"], checkpoint_path=pipeline_config["checkpointing"]["path"])
        else:
            checkpoint_info = CheckpointInfo(checkpoint_interval=0, checkpoint_path="")

        # TODO(#127): Optionally transfer random seed
        req = RegisterTrainServerRequest(
            training_id=pipeline_id,  # TODO(#74): The trainer needs to switch to the pipeline/trigger model
            model_id=pipeline_config["model"]["id"],
            batch_size=pipeline_config["training"]["batch_size"],
            torch_optimizer="SGD",  # TODO(MaxiBoether): Support Optimizer/Criterion in pipeline config
            torch_criterion="CrossEntropyLoss",  # TODO(MaxiBoether): Support Optimizer/Criterion in pipeline config
            # TODO(MaxiBoether): Support Optimizer/Criterion in pipeline config
            criterion_parameters=JsonString(value="{}"),
            # TODO(MaxiBoether): Support Optimizer/Criterion in pipeline config
            optimizer_parameters=JsonString(value=optimizer_parameters),
            model_configuration=JsonString(value=model_config),
            data_info=Data(
                dataset_id=pipeline_config["data"]["dataset_id"],
                num_dataloaders=pipeline_config["training"]["dataloader_workers"],
            ),
            checkpoint_info=checkpoint_info,
            transform_list=transform_list,
            bytes_parser=PythonString(value=pipeline_config["data"]["bytes_parser_function"])
        )

        response: RegisterTrainServerResponse = self.trainer_server.register(req)

        if not response.success:
            raise RuntimeError("Registration at trainer did go wrong: {response}")

        # TODO(#74): Return training ID from trainer server here
        return 42

    def start_training(self, pipeline_id: int, trigger_id: int, pipeline_config: dict) -> int:
        if not self.connected_to_trainer_server:
            raise ConnectionError("Tried to start training at trainer server, but not there is no gRPC connection..")

        training_id = self._register_training(pipeline_id, trigger_id, pipeline_config)
        logger.info(f"Registered training at trainer server with id {training_id}")

        req = StartTrainingRequest(
            training_id=training_id,
            device=pipeline_config["training"]["device"],
            train_until_sample_id="new",  # TODO(#74): Remove this
            load_checkpoint_path="", # TODO(#127): Make pretrained model optional in protos + transfer bytes of initial model instead. 
        )

        response: StartTrainingResponse = self.trainer_server.start_training(req)

        if not response.training_started:
            raise RuntimeError("Starting training at trainer did go wrong: {response}")

        logger.info(f"Started training {training_id} at trainer server.")

        return training_id

    def wait_for_training_completion(self, training_id: int) -> None:
        if not self.connected_to_trainer_server:
            raise ConnectionError(
                "Tried to wait for training to finish at trainer server, but not there is no gRPC connection."
            )

        while True:
            req = TrainingStatusRequest(training_id=training_id)
            res: TrainingStatusResponse = self.trainer_server.get_training_status(req)

            if not res.valid:
                raise RuntimeError(f"Training {training_id} is invalid at server: {res}\n")

            if res.is_running:
                emoji = "⏳"
            else:
                emoji = "✅"

            if res.blocked:
                logger.warning("Trainer Server returned a blocked response: {res}\n")
            else:
                if res.state_available:
                    # TODO(create issue): return total number of batches/samples
                    logger.info(f"\r{emoji} Batch {res.batches_seen}/?, Sample {res.samples_seen}/?")
                else:
                    logger.warning(
                        "Trainer server is not blocked, but no state is available. This might be because we queried status for a finished training.\n"
                    )

            if res.is_running:
                sleep(3)
            else:
                break

        logger.info("Training completed 🚀")

    # pylint: disable-next=unused-argument
    def get_trained_model(self, training_id: int) -> bytes:
        # TODO(create issue): Implement at trainer.
        return b""
