# pylint: disable=no-name-in-module
import json
import logging
from time import sleep
from typing import Iterable, Optional

import enlighten
import grpc
from modyn.selector.internal.grpc.generated.selector_pb2 import (
    DataInformRequest,
    GetNumberOfSamplesRequest,
    GetStatusBarScaleRequest,
)
from modyn.selector.internal.grpc.generated.selector_pb2 import JsonString as SelectorJsonString
from modyn.selector.internal.grpc.generated.selector_pb2 import (
    NumberOfSamplesResponse,
    RegisterPipelineRequest,
    SeedSelectorRequest,
    StatusBarScaleResponse,
    TriggerResponse,
)
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.storage.internal.grpc.generated import storage_pb2
from modyn.storage.internal.grpc.generated.storage_pb2 import (
    DatasetAvailableRequest,
    GetCurrentTimestampResponse,
    GetDataInIntervalRequest,
    GetDataInIntervalResponse,
    GetNewDataSinceRequest,
    GetNewDataSinceResponse,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.supervisor.internal.supervisor_counter import SupervisorCounter
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import CheckpointInfo, Data
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import JsonString as TrainerServerJsonString
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import (
    PythonString,
    StartTrainingRequest,
    StartTrainingResponse,
    StoreFinalModelRequest,
    StoreFinalModelResponse,
    TrainerAvailableRequest,
    TrainerAvailableResponse,
    TrainingStatusRequest,
    TrainingStatusResponse,
)
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2_grpc import TrainerServerStub
from modyn.utils import MAX_MESSAGE_SIZE, grpc_connection_established

logger = logging.getLogger(__name__)


class GRPCHandler:
    # pylint: disable=too-many-instance-attributes

    def __init__(self, modyn_config: dict, progress_mgr: enlighten.Manager, status_bar: enlighten.StatusBar):
        self.config = modyn_config
        self.connected_to_storage = False
        self.connected_to_trainer_server = False
        self.connected_to_selector = False
        self.progress_mgr = progress_mgr
        self.status_bar = status_bar

        self.init_storage()
        self.init_selector()
        self.init_trainer_server()

    def init_storage(self) -> None:
        assert self.config is not None
        storage_address = f"{self.config['storage']['hostname']}:{self.config['storage']['port']}"
        self.storage_channel = grpc.insecure_channel(
            storage_address,
            options=[
                ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
                ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ],
        )

        if not grpc_connection_established(self.storage_channel):
            raise ConnectionError(f"Could not establish gRPC connection to storage at {storage_address}.")

        self.storage = StorageStub(self.storage_channel)
        logger.info("Successfully connected to storage.")
        self.connected_to_storage = True

    def init_selector(self) -> None:
        assert self.config is not None
        selector_address = f"{self.config['selector']['hostname']}:{self.config['selector']['port']}"
        self.selector_channel = grpc.insecure_channel(
            selector_address,
            options=[
                ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
                ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ],
        )

        if not grpc_connection_established(self.selector_channel):
            raise ConnectionError(f"Could not establish gRPC connection to selector at {selector_address}.")

        self.selector = SelectorStub(self.selector_channel)
        logger.info("Successfully connected to selector.")
        self.connected_to_selector = True

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

    def get_new_data_since(self, dataset_id: str, timestamp: int) -> Iterable[list[tuple[int, int, int]]]:
        if not self.connected_to_storage:
            raise ConnectionError("Tried to fetch data from storage, but no connection was made.")

        request = GetNewDataSinceRequest(dataset_id=dataset_id, timestamp=timestamp)
        response: GetNewDataSinceResponse
        for response in self.storage.GetNewDataSince(request):
            data = list(zip(response.keys, response.timestamps, response.labels))
            yield data

    def get_data_in_interval(
        self, dataset_id: str, start_timestamp: int, end_timestamp: int
    ) -> Iterable[list[tuple[int, int, int]]]:
        if not self.connected_to_storage:
            raise ConnectionError("Tried to fetch data from storage, but no connection was made.")

        request = GetDataInIntervalRequest(
            dataset_id=dataset_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )
        response: GetDataInIntervalResponse
        for response in self.storage.GetDataInInterval(request):
            data = list(zip(response.keys, response.timestamps, response.labels))
            yield data

    def get_time_at_storage(self) -> int:
        if not self.connected_to_storage:
            raise ConnectionError("Tried to fetch data from storage, but no connection was made.")

        response: GetCurrentTimestampResponse = self.storage.GetCurrentTimestamp(
            storage_pb2.google_dot_protobuf_dot_empty__pb2.Empty()  # type: ignore
        )

        return response.timestamp

    def register_pipeline_at_selector(self, pipeline_config: dict) -> int:
        if not self.connected_to_selector:
            raise ConnectionError("Tried to register pipeline at selector, but no connection was made.")

        pipeline_id = self.selector.register_pipeline(
            RegisterPipelineRequest(
                num_workers=pipeline_config["training"]["dataloader_workers"],
                selection_strategy=SelectorJsonString(
                    value=json.dumps(pipeline_config["training"]["selection_strategy"])
                ),
            )
        ).pipeline_id

        logger.info(f"Registered pipeline {pipeline_config['pipeline']['name']} at selector with ID {pipeline_id}")
        return pipeline_id

    # pylint: disable-next=unused-argument
    def unregister_pipeline_at_selector(self, pipeline_id: int) -> None:
        #  # TODO(#64,#124): Implement.
        pass

    def inform_selector(self, pipeline_id: int, data: list[tuple[int, int, int]]) -> None:
        keys, timestamps, labels = zip(*data)
        request = DataInformRequest(pipeline_id=pipeline_id, keys=keys, timestamps=timestamps, labels=labels)
        self.selector.inform_data(request)

    def inform_selector_and_trigger(self, pipeline_id: int, data: list[tuple[int, int, int]]) -> int:
        keys: list[int]
        timestamps: list[int]
        labels: list[int]
        if len(data) == 0:
            keys, timestamps, labels = [], [], []
        else:
            # mypy fails to recognize that this is correct
            keys, timestamps, labels = zip(*data)  # type: ignore

        request = DataInformRequest(pipeline_id=pipeline_id, keys=keys, timestamps=timestamps, labels=labels)
        response: TriggerResponse = self.selector.inform_data_and_trigger(request)

        trigger_id = response.trigger_id
        logging.info(f"Informed selector about trigger. Got trigger id {trigger_id}.")

        return trigger_id

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

    # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    def start_training(
        self, pipeline_id: int, trigger_id: int, pipeline_config: dict, previous_model_id: Optional[int]
    ) -> int:
        if not self.connected_to_trainer_server:
            raise ConnectionError("Tried to start training at trainer server, but not there is no gRPC connection.")

        if "config" in pipeline_config["model"]:
            model_config = json.dumps(pipeline_config["model"]["config"])
        else:
            model_config = "{}"

        optimizers_config = {}
        for optimizer in pipeline_config["training"]["optimizers"]:
            optimizer_config = {}
            optimizer_config["algorithm"] = optimizer["algorithm"]
            optimizer_config["source"] = optimizer["source"]
            optimizer_config["param_groups"] = []
            for param_group in optimizer["param_groups"]:
                config_dict = param_group["config"] if "config" in param_group else {}
                optimizer_config["param_groups"].append({"module": param_group["module"], "config": config_dict})
            optimizers_config[optimizer["name"]] = optimizer_config

        lr_scheduler_configs = {}
        if "lr_scheduler" in pipeline_config["training"]:
            lr_scheduler_configs = pipeline_config["training"]["lr_scheduler"]
            if "config" not in lr_scheduler_configs:
                lr_scheduler_configs["config"] = {}

        if "config" in pipeline_config["training"]["optimization_criterion"]:
            criterion_config = json.dumps(pipeline_config["training"]["optimization_criterion"]["config"])
        else:
            criterion_config = "{}"

        if "epochs_per_trigger" in pipeline_config["training"]:
            epochs_per_trigger = pipeline_config["training"]["epochs_per_trigger"]
        else:
            epochs_per_trigger = 1

        if "seed" in pipeline_config["training"]:
            seed = pipeline_config["training"]["seed"]
        else:
            seed = None

        if "tokenizer" in pipeline_config["data"]:
            tokenizer = pipeline_config["data"]["tokenizer"]
        else:
            tokenizer = None

        if "transformations" in pipeline_config["data"]:
            transform_list = pipeline_config["data"]["transformations"]
        else:
            transform_list = []

        if "label_transformer_function" in pipeline_config["data"]:
            label_transformer = pipeline_config["data"]["label_transformer_function"]
        else:
            label_transformer = ""

        if pipeline_config["training"]["checkpointing"]["activated"]:
            if (
                "interval" not in pipeline_config["training"]["checkpointing"]
                or "path" not in pipeline_config["training"]["checkpointing"]
            ):
                raise ValueError("Checkpointing is enabled, but interval or path not given.")

            checkpoint_info = CheckpointInfo(
                checkpoint_interval=pipeline_config["training"]["checkpointing"]["interval"],
                checkpoint_path=pipeline_config["training"]["checkpointing"]["path"],
            )
        else:
            checkpoint_info = CheckpointInfo(checkpoint_interval=0, checkpoint_path="")

        amp = pipeline_config["training"]["amp"] if "amp" in pipeline_config["training"] else False

        if "grad_scaler_config" in pipeline_config["training"]:
            grad_scaler_config = pipeline_config["training"]["grad_scaler_config"]
        else:
            grad_scaler_config = {}

        start_training_kwargs = {
            "pipeline_id": pipeline_id,
            "trigger_id": trigger_id,
            "device": pipeline_config["training"]["device"],
            "amp": amp,
            "model_id": pipeline_config["model"]["id"],
            "model_configuration": TrainerServerJsonString(value=model_config),
            "use_pretrained_model": previous_model_id is not None,
            "pretrained_model_id": previous_model_id or -1,
            "load_optimizer_state": False,  # TODO(#137): Think about this.
            "batch_size": pipeline_config["training"]["batch_size"],
            "torch_optimizers_configuration": TrainerServerJsonString(value=json.dumps(optimizers_config)),
            "torch_criterion": pipeline_config["training"]["optimization_criterion"]["name"],
            "criterion_parameters": TrainerServerJsonString(value=criterion_config),
            "data_info": Data(
                dataset_id=pipeline_config["data"]["dataset_id"],
                num_dataloaders=pipeline_config["training"]["dataloader_workers"],
            ),
            "checkpoint_info": checkpoint_info,
            "transform_list": transform_list,
            "bytes_parser": PythonString(value=pipeline_config["data"]["bytes_parser_function"]),
            "label_transformer": PythonString(value=label_transformer),
            "lr_scheduler": TrainerServerJsonString(value=json.dumps(lr_scheduler_configs)),
            "grad_scaler_configuration": TrainerServerJsonString(value=json.dumps(grad_scaler_config)),
            "epochs_per_trigger": epochs_per_trigger,
            "seed": seed,
            "tokenizer": PythonString(value=tokenizer) if tokenizer is not None else None,
        }

        cleaned_kwargs = {k: v for k, v in start_training_kwargs.items() if v is not None}

        req = StartTrainingRequest(**cleaned_kwargs)

        response: StartTrainingResponse = self.trainer_server.start_training(req)

        if not response.training_started:
            raise RuntimeError(f"Starting training at trainer did go wrong: {response}")

        training_id = response.training_id
        logger.info(f"Started training {training_id} at trainer server.")

        return training_id

    def get_number_of_samples(self, pipeline_id: int, trigger_id: int) -> int:
        request = GetNumberOfSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id)
        response: NumberOfSamplesResponse = self.selector.get_number_of_samples(request)

        return response.num_samples

    def get_status_bar_scale(self, pipeline_id: int) -> int:
        request = GetStatusBarScaleRequest(pipeline_id=pipeline_id)
        response: StatusBarScaleResponse = self.selector.get_status_bar_scale(request)

        return response.status_bar_scale

    # pylint: disable=too-many-nested-blocks

    def wait_for_training_completion(
        self, training_id: int, pipeline_id: int, trigger_id: int
    ) -> None:  # pragma: no cover
        if not self.connected_to_trainer_server:
            raise ConnectionError(
                "Tried to wait for training to finish at trainer server, but not there is no gRPC connection."
            )
        self.status_bar.update(demo=f"Waiting for training (id = {training_id})")

        total_samples = self.get_number_of_samples(pipeline_id, trigger_id)
        status_bar_scale = self.get_status_bar_scale(pipeline_id)

        sample_pbar = SupervisorCounter(self.progress_mgr, training_id, total_samples, status_bar_scale)

        blocked_in_a_row = 0

        while True:
            req = TrainingStatusRequest(training_id=training_id)
            res: TrainingStatusResponse = self.trainer_server.get_training_status(req)

            if not res.valid:
                raise RuntimeError(f"Training {training_id} is invalid at server:\n{res}\n")

            if res.blocked:
                blocked_in_a_row += 1

                if blocked_in_a_row >= 3:
                    logger.warning(
                        f"Trainer Server returned {blocked_in_a_row} blocked responses in a row, cannot update status."
                    )

            else:
                if res.HasField("exception") and res.exception is not None:
                    raise RuntimeError(f"Exception at trainer server occured during training:\n{res.exception}\n\n")

                blocked_in_a_row = 0

                if res.state_available:
                    assert (res.HasField("samples_seen") and res.HasField("batches_seen")) or (
                        res.HasField("downsampling_samples_seen") and res.HasField("downsampling_batches_seen")
                    ), f"Inconsistent server response:\n{res}"

                    sample_pbar.progress_counter(res.samples_seen, res.downsampling_samples_seen, res.is_training)

                elif res.is_running:
                    logger.warning("Trainer server is not blocked and running, but no state is available.")

            if res.is_running:
                sleep(2)
            else:
                break

        sample_pbar.close_counter()
        logger.info("Training completed 🚀")

    def store_trained_model(self, training_id: int) -> int:
        logger.info(f"Storing trained model for training {training_id}")

        req = StoreFinalModelRequest(training_id=training_id)
        res: StoreFinalModelResponse = self.trainer_server.store_final_model(req)

        if not res.valid_state:
            raise RuntimeError(
                f"Cannot fetch trained model for training {training_id}"
                + " since training is invalid or training still running"
            )

        logger.info(f"Model {res.model_id} has been stored successfully")

        return res.model_id

    def seed_selector(self, seed: int) -> None:
        if not (0 <= seed <= 100 and isinstance(seed, int)):
            raise ValueError("The seed must be an integer in [0,100]")
        if not self.connected_to_selector:
            raise ConnectionError("Tried to seed the selector, but no connection was made.")

        success = self.selector.seed_selector(
            SeedSelectorRequest(
                seed=seed,
            ),
        ).success

        assert success, "Something went wrong while seeding the selector"
