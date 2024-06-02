import contextlib
import datetime
import json
import logging
import multiprocessing as mp
import os
import socket
import time
from concurrent import futures
from typing import Any, Callable, Generator, Optional

import grpc

# pylint: disable=no-name-in-module
from modyn.supervisor.internal.utils import TrainingStatusReporter
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
from modyn.utils import MAX_MESSAGE_SIZE, grpc_common_config, grpc_connection_established

logger = logging.getLogger(__name__)

# Minimum 2 processes and 4 threads per process, currently max 64 processes
CPU_CORES = os.cpu_count()
if CPU_CORES is None:  # cannot do that in single expression due to mypy...
    CPU_CORES = 64
NUM_GPRC_PROCESSES = max(2, min(64, CPU_CORES))
PROCESS_THREAD_WORKERS = max(4, int(NUM_GPRC_PROCESSES / 4))


@contextlib.contextmanager
def reserve_port(port: str) -> Generator:
    """Find and reserve a port for all subprocesses to use."""
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
        raise RuntimeError("Failed to set SO_REUSEPORT.")
    sock.bind(("", int(port)))
    try:
        assert sock.getsockname()[1] == int(port)
        yield port
    finally:
        sock.close()


def _wait_forever(server: Any) -> None:
    try:
        while True:
            time.sleep(datetime.timedelta(days=1).total_seconds())
    except KeyboardInterrupt:
        server.stop(None)


def _run_server_worker(
    bind_address: str, add_servicer_callback: Callable, modyn_config: dict, callback_kwargs: dict
) -> None:
    """Start a server in a subprocess."""
    logging.info(f"[{os.getpid()}] Starting new gRPC server process.")

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=PROCESS_THREAD_WORKERS,
        ),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ("grpc.so_reuseport", 1),
        ],
    )

    add_servicer_callback(modyn_config, server, **callback_kwargs)
    server.add_insecure_port(bind_address)
    server.start()
    _wait_forever(server)


class GenericGRPCServer:
    def __init__(
        self, modyn_config: dict, port: str, add_servicer_callback: Callable, callback_kwargs: Optional[dict] = None
    ) -> None:
        """Initialize the GRPC server."""
        self.port = port
        self.modyn_config = modyn_config
        self.add_servicer_callback = add_servicer_callback
        self.callback_kwargs = callback_kwargs if callback_kwargs is not None else {}
        self.workers: list[mp.Process] = []

    def __enter__(self) -> Any:
        """Enter the context manager.

        Returns:
            grpc.Server: GRPC server
        """
        logger.info(f"[{os.getpid()}] Starting server. Listening on port {self.port}")
        with reserve_port(self.port) as port:
            bind_address = "[::]:" + port
            for _ in range(NUM_GPRC_PROCESSES):
                worker = mp.Process(
                    target=_run_server_worker,
                    args=(bind_address, self.add_servicer_callback, self.modyn_config, self.callback_kwargs),
                )
                worker.start()
                self.workers.append(worker)

        return self

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["add_servicer_callback"]
        return state

    def wait_for_termination(self) -> None:
        for worker in self.workers:
            worker.join()

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Exception) -> None:
        """Exit the context manager.

        Args:
            exc_type (type): exception type
            exc_val (Exception): exception value
            exc_tb (Exception): exception traceback
        """
        self.wait_for_termination()
        del self.workers


class TrainerServerGRPCHandlerMixin:
    def __init__(
        self,
        modyn_config: dict,
        training_status_queue: Optional[mp.Queue] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = modyn_config
        self.training_status_queue = training_status_queue
        self.connected_to_trainer_server = False
        self.trainer_server: Optional[TrainerServerStub] = None
        self.trainer_server_channel: Optional[grpc.Channel] = None

    def init_trainer_server(self) -> None:
        assert self.config is not None
        trainer_server_address = f"{self.config['trainer_server']['hostname']}:{self.config['trainer_server']['port']}"
        self.trainer_server_channel = grpc.insecure_channel(trainer_server_address, options=grpc_common_config())

        if not grpc_connection_established(self.trainer_server_channel):
            raise ConnectionError(f"Could not establish gRPC connection to trainer server at {trainer_server_address}.")

        self.trainer_server = TrainerServerStub(self.trainer_server_channel)
        logger.info("Successfully connected to trainer server.")
        self.connected_to_trainer_server = self.trainer_server is not None

    def trainer_server_available(self) -> bool:
        assert self.trainer_server is not None

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
        self,
        pipeline_id: int,
        trigger_id: int,
        pipeline_config: dict,
        previous_model_id: Optional[int],
        num_samples_to_pass: Optional[int] = None,
    ) -> int:
        assert self.trainer_server is not None
        if not self.connected_to_trainer_server:
            raise ConnectionError("Tried to start training at trainer server, but not there is no gRPC connection.")

        optimizers_config = {}
        for optimizer in pipeline_config["training"]["optimizers"]:
            optimizer_config = {"algorithm": optimizer["algorithm"], "source": optimizer["source"], "param_groups": []}
            for param_group in optimizer["param_groups"]:
                config_dict = param_group.get("config") or {}
                optimizer_config["param_groups"].append({"module": param_group["module"], "config": config_dict})
            optimizers_config[optimizer["name"]] = optimizer_config

        lr_scheduler_configs = {}
        if pipeline_config["training"].get("lr_scheduler"):
            lr_scheduler_configs = pipeline_config["training"]["lr_scheduler"]
            lr_scheduler_configs["config"] = lr_scheduler_configs.get("config") or {}

        criterion_config = json.dumps(pipeline_config["training"]["optimization_criterion"].get("config") or {})

        epochs_per_trigger = pipeline_config["training"].get("epochs_per_trigger") or 1

        if pipeline_config["training"].get("num_prefetched_partitions"):
            num_prefetched_partitions = pipeline_config["training"]["num_prefetched_partitions"]
        else:
            if pipeline_config["training"].get("prefetched_partitions"):
                raise ValueError(
                    "Found `prefetched_partitions` instead of `num_prefetched_partitions`in training configuration."
                    + " Please rename/remove that configuration"
                )
            logger.warning("Number of prefetched partitions not explicitly given in training config - defaulting to 1.")
            num_prefetched_partitions = 1

        if pipeline_config["training"].get("parallel_prefetch_requests"):
            parallel_prefetch_requests = pipeline_config["training"]["parallel_prefetch_requests"]
        else:
            logger.warning(
                "Number of parallel prefetch requests not explicitly given in training config - defaulting to 1."
            )
            parallel_prefetch_requests = 1

        seed = pipeline_config["training"].get("seed", None)
        tokenizer = pipeline_config["data"].get("tokenizer", None)
        transform_list = pipeline_config["data"].get("transformations") or []
        label_transformer = pipeline_config["data"].get("label_transformer_function") or ""

        if pipeline_config["training"]["checkpointing"]["activated"]:
            if (
                pipeline_config["training"]["checkpointing"].get("interval") is None
                or pipeline_config["training"]["checkpointing"].get("path") is None
            ):
                raise ValueError("Checkpointing is enabled, but interval or path not given.")

            checkpoint_info = CheckpointInfo(
                checkpoint_interval=pipeline_config["training"]["checkpointing"]["interval"],
                checkpoint_path=pipeline_config["training"]["checkpointing"]["path"],
            )
        else:
            checkpoint_info = CheckpointInfo(checkpoint_interval=0, checkpoint_path="")

        grad_scaler_config = pipeline_config["training"].get("grad_scaler_config") or {}

        start_training_kwargs = {
            "pipeline_id": pipeline_id,
            "trigger_id": trigger_id,
            "device": pipeline_config["training"]["device"],
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
            "num_prefetched_partitions": num_prefetched_partitions,
            "parallel_prefetch_requests": parallel_prefetch_requests,
            "seed": seed,
            "tokenizer": PythonString(value=tokenizer) if tokenizer is not None else None,
            "num_samples_to_pass": num_samples_to_pass,
        }

        cleaned_kwargs = {k: v for k, v in start_training_kwargs.items() if v is not None}

        req = StartTrainingRequest(**cleaned_kwargs)

        response: StartTrainingResponse = self.trainer_server.start_training(req)

        if not response.training_started:
            raise RuntimeError(f"Starting training at trainer did go wrong: {response}")

        training_id = response.training_id
        logger.info(f"Started training {training_id} at trainer server.")

        return training_id

    # pylint: disable=too-many-nested-blocks
    def wait_for_training_completion(
        self, training_id: int, training_reporter: TrainingStatusReporter
    ) -> dict[str, Any]:  # pragma: no cover
        assert self.training_status_queue is not None
        assert self.trainer_server is not None
        if not self.connected_to_trainer_server:
            raise ConnectionError(
                "Tried to wait for training to finish at trainer server, but not there is no gRPC connection."
            )
        logger.debug("wait for training completion")
        training_reporter.create_tracker()
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
                    raise RuntimeError(f"Exception at trainer server occurred during training:\n{res.exception}\n\n")

                blocked_in_a_row = 0

                if res.state_available:
                    assert (res.HasField("samples_seen") and res.HasField("batches_seen")) or (
                        res.HasField("downsampling_samples_seen") and res.HasField("downsampling_batches_seen")
                    ), f"Inconsistent server response:\n{res}"

                    training_reporter.progress_counter(res.samples_seen, res.downsampling_samples_seen, res.is_training)

                elif res.is_running:
                    logger.warning("Trainer server is not blocked and running, but no state is available.")

            if res.is_running:
                time.sleep(2)
            else:
                trainer_log = json.loads(res.log.value)
                break

        training_reporter.close_counter()

        return trainer_log

    def store_trained_model(self, training_id: int) -> int:
        assert self.trainer_server is not None

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
