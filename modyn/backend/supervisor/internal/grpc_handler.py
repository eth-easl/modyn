import json
import logging
import pathlib
from time import sleep
from typing import Optional

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
    StartTrainingRequest,
    StartTrainingResponse,
    TrainerAvailableRequest,
    TrainerAvailableResponse,
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

        return list(zip(response.keys, response.timestamps))

    def get_data_in_interval(self, dataset_id: str, start_timestamp: int, end_timestamp: int) -> list[tuple[str, int]]:
        if not self.connected_to_storage:
            raise ConnectionError("Tried to fetch data from storage, but no connection was made.")

        request = GetDataInIntervalRequest(
            dataset_id=dataset_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )
        response: GetDataInIntervalResponse = self.storage.GetDataInInterval(request)

        return list(zip(response.keys, response.timestamps))

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
        #  # TODO(#64,#124): Implement.
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
        response: TrainerAvailableResponse = self.trainer_server.trainer_available(request)

        logger.info(f"Trainer Server Availability = {response.available}")

        return response.available

    # pylint: disable-next=unused-argument
    def stop_training_at_trainer_server(self, training_id: int) -> None:
        # TODO(#130): Implement this at trainer server.
        logger.error("The trainer server currently does not support remotely stopping training, ignoring.")

    # pylint: disable=too-many-branches,too-many-locals
    def start_training(
        self, pipeline_id: int, trigger_id: int, pipeline_config: dict, previous_model: Optional[pathlib.Path]
    ) -> int:
        if not self.connected_to_trainer_server:
            raise ConnectionError("Tried to start training at trainer server, but not there is no gRPC connection.")

        if "config" in pipeline_config["model"]:
            model_config = json.dumps(pipeline_config["model"]["config"])
        else:
            model_config = "{}"

        if previous_model is not None:
            use_pretrained_model = True
            with open(previous_model, "rb") as file:
                pretrained_model = file.read()
        else:
            use_pretrained_model = False
            pretrained_model = None

        if "config" in pipeline_config["optimizer"]:
            optimizer_config = json.dumps(pipeline_config["optimizer"]["config"])
        else:
            optimizer_config = "{}"

        if "config" in pipeline_config["optimization_criterion"]:
            criterion_config = json.dumps(pipeline_config["optimization_criterion"]["config"])
        else:
            criterion_config = "{}"

        if "transformations" in pipeline_config["data"]:
            transform_list = pipeline_config["data"]["transformations"]
        else:
            transform_list = []

        if pipeline_config["checkpointing"]["activated"]:
            if "interval" not in pipeline_config["checkpointing"] or "path" not in pipeline_config["checkpointing"]:
                raise ValueError("Checkpointing is enabled, but interval or path not given.")

            checkpoint_info = CheckpointInfo(
                checkpoint_interval=pipeline_config["checkpointing"]["interval"],
                checkpoint_path=pipeline_config["checkpointing"]["path"],
            )
        else:
            checkpoint_info = CheckpointInfo(checkpoint_interval=0, checkpoint_path="")

        req = StartTrainingRequest(
            pipeline_id=pipeline_id,
            training_id=trigger_id,
            device=pipeline_config["training"]["device"],
            model_id=pipeline_config["model"]["id"],
            model_configuration=JsonString(value=model_config),
            use_pretrained_model=use_pretrained_model,
            pretrained_model=pretrained_model,
            batch_size=pipeline_config["training"]["batch_size"],
            torch_optimizer=pipeline_config["optimizer"]["name"],
            optimizer_parameters=JsonString(value=optimizer_config),
            torch_criterion=pipeline_config["optimization_criterion"]["name"],
            criterion_parameters=JsonString(value=criterion_config),
            data_info=Data(
                dataset_id=pipeline_config["data"]["dataset_id"],
                num_dataloaders=pipeline_config["training"]["dataloader_workers"],
            ),
            checkpoint_info=checkpoint_info,
            transform_list=transform_list,
            bytes_parser=PythonString(value=pipeline_config["data"]["bytes_parser_function"]),
        )

        response: StartTrainingResponse = self.trainer_server.start_training(req)

        if not response.training_started:
            raise RuntimeError(f"Starting training at trainer did go wrong: {response}")

        training_id = response.training_id
        logger.info(f"Started training {training_id} at trainer server.")

        return training_id

    def wait_for_training_completion(self, training_id: int) -> None:  # pragma: no cover
        if not self.connected_to_trainer_server:
            raise ConnectionError(
                "Tried to wait for training to finish at trainer server, but not there is no gRPC connection."
            )

        while True:
            req = TrainingStatusRequest(training_id=training_id)
            res: TrainingStatusResponse = self.trainer_server.get_training_status(req)

            if not res.valid:
                raise RuntimeError(f"Training {training_id} is invalid at server: {res}\n")

            if res.blocked:
                logger.warning("Trainer Server returned a blocked response: {res}\n")
            else:
                if res.exception is not None:
                    logger.error(f"Exception occured during training: {res.exception}\n\n{res}\n")

                if res.state_available:
                    logger.info(
                        f"\r{'⏳' if res.is_running else '✅'} Batch {res.batches_seen}/{res.batches_total}"
                        + f"Sample {res.samples_seen}/{res.samples_total}"
                    )
                else:
                    logger.warning(
                        "Trainer server is not blocked, but no state is available. "
                        "This might be because we queried status for a finished training.\n"
                    )

            if res.is_running:
                sleep(3)
            else:
                break

        logger.info("Training completed 🚀")

    def fetch_trained_model(self, training_id: int, storage_dir: pathlib.Path) -> pathlib.Path:
        logger.info(f"Fetching trained model for training {training_id}")
        req = TrainingStatusRequest(training_id=training_id)
        res: TrainingStatusResponse = self.trainer_server.get_training_status(req)

        if not res.valid:
            raise RuntimeError(f"Cannot fetch trained model for training {training_id} since training is invalid")

        if res.is_running:
            raise RuntimeError(
                f"Cannot fetch trained model for training {training_id} since training has not finished yet"
            )

        model = b""  # TODO(#74): fetch final model from trainer

        model_path = storage_dir / f"{training_id}.modyn"
        logger.info(f"Fetched model, storing at {model_path}")

        with open(model_path, "wb") as file:
            file.write(model)

        logger.info("Wrote model to disk.")

        return model_path
