import contextlib
import datetime
import json
import logging
import multiprocessing as mp
import os
import socket
import time
from collections.abc import Callable, Generator
from concurrent import futures
from typing import Any

import grpc

from modyn.config.schema.pipeline import TrainingConfig
from modyn.config.schema.pipeline.data import DataConfig

# pylint: disable=no-name-in-module
from modyn.supervisor.internal.utils import TrainingStatusReporter
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import (
    CheckpointInfo,
    Data,
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
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import JsonString as TrainerServerJsonString
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
        self, modyn_config: dict, port: str, add_servicer_callback: Callable, callback_kwargs: dict | None = None
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
    def __init__(self, modyn_config: dict, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = modyn_config
        self.connected_to_trainer_server = False
        self.trainer_server: TrainerServerStub | None = None
        self.trainer_server_channel: grpc.Channel | None = None

    def init_trainer_server(self) -> None:
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

    # pylint: disable=too-many-locals, too-many-branches
    @staticmethod
    def prepare_start_training_request(
        pipeline_id: int,
        trigger_id: int,
        training_config: TrainingConfig,
        data_config: DataConfig,
        previous_model_id: int | None,
        num_samples_to_pass: int | None = None,
    ) -> StartTrainingRequest:
        optimizers_config = {}
        for optimizer in training_config.optimizers:
            optimizer_config: dict[str, Any] = {
                "algorithm": optimizer.algorithm,
                "source": optimizer.source,
                "param_groups": [],
            }
            for param_group in optimizer.param_groups:
                optimizer_config["param_groups"].append({"module": param_group.module, "config": param_group.config})
            optimizers_config[optimizer.name] = optimizer_config

        lr_scheduler_configs: dict[str, Any] = {}
        if training_config.lr_scheduler is not None:
            lr_scheduler_configs = training_config.lr_scheduler.model_dump(by_alias=True)

        criterion_config = json.dumps(training_config.optimization_criterion.config)
        tokenizer = data_config.tokenizer

        if training_config.checkpointing.activated:
            # the None-ility of the two fields are already validated by pydantic
            checkpoint_info = CheckpointInfo(
                checkpoint_interval=training_config.checkpointing.interval,  # type: ignore[arg-type]
                checkpoint_path=str(training_config.checkpointing.path),  # type: ignore[arg-type]
            )
        else:
            checkpoint_info = CheckpointInfo(checkpoint_interval=0, checkpoint_path="")

        grad_scaler_config = training_config.grad_scaler_config if training_config.grad_scaler_config else {}

        return StartTrainingRequest(
            pipeline_id=pipeline_id,
            trigger_id=trigger_id,
            device=training_config.device,
            use_pretrained_model=previous_model_id is not None,
            pretrained_model_id=previous_model_id or -1,
            load_optimizer_state=False,  # TODO(#137): Think about this.
            batch_size=training_config.batch_size,
            torch_optimizers_configuration=TrainerServerJsonString(value=json.dumps(optimizers_config)),
            torch_criterion=training_config.optimization_criterion.name,
            criterion_parameters=TrainerServerJsonString(value=criterion_config),
            data_info=Data(
                dataset_id=data_config.dataset_id,
                num_dataloaders=training_config.dataloader_workers,
            ),
            checkpoint_info=checkpoint_info,
            transform_list=data_config.transformations,
            bytes_parser=PythonString(value=data_config.bytes_parser_function),
            label_transformer=PythonString(value=data_config.label_transformer_function),
            lr_scheduler=TrainerServerJsonString(value=json.dumps(lr_scheduler_configs)),
            grad_scaler_configuration=TrainerServerJsonString(value=json.dumps(grad_scaler_config)),
            epochs_per_trigger=training_config.epochs_per_trigger,
            num_prefetched_partitions=training_config.num_prefetched_partitions,
            parallel_prefetch_requests=training_config.parallel_prefetch_requests,
            seed=training_config.seed,  # seed is an optional field which can accept None
            # tokenizer is an optional field which can accept None
            tokenizer=PythonString(value=tokenizer) if tokenizer is not None else None,
            num_samples_to_pass=num_samples_to_pass if num_samples_to_pass is not None else 0,
            shuffle=training_config.shuffle,
            enable_accurate_gpu_measurements=training_config.enable_accurate_gpu_measurements,
            record_loss_every=training_config.record_loss_every,
            drop_last_batch=training_config.drop_last_batch,
            generative=training_config.generative,
            grad_norm=training_config.grad_norm if training_config.grad_norm != 0.0 else None,
            lora=training_config.lora,
            kadapter=training_config.kadapter,
            prompt_tuning=training_config.prompt_tuning,
            prefix_tuning=training_config.prefix_tuning,
        )

    def start_training(
        self,
        pipeline_id: int,
        trigger_id: int,
        training_config: TrainingConfig,
        data_config: DataConfig,
        previous_model_id: int | None,
        num_samples_to_pass: int | None = None,
    ) -> int:
        assert self.trainer_server is not None
        if not self.connected_to_trainer_server:
            raise ConnectionError("Tried to start training at trainer server, but not there is no gRPC connection.")

        req = self.prepare_start_training_request(
            pipeline_id,
            trigger_id,
            training_config,
            data_config,
            previous_model_id,
            num_samples_to_pass,
        )
        response: StartTrainingResponse = self.trainer_server.start_training(req)

        if not response.training_started:
            raise RuntimeError(f"Starting training at trainer did go wrong: {response}")

        training_id = response.training_id
        logger.info(f"Started training {training_id} at trainer server.")

        return training_id

    # pylint: disable=too-many-nested-blocks
    def wait_for_training_completion(
        self, training_id: int, training_reporter: TrainingStatusReporter | None = None
    ) -> dict[str, Any]:
        assert self.trainer_server is not None
        if not self.connected_to_trainer_server:
            raise ConnectionError(
                "Tried to wait for training to finish at trainer server, but not there is no gRPC connection."
            )
        logger.debug("wait for training completion")
        if training_reporter is not None:
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

                    if training_reporter is not None:
                        training_reporter.progress_counter(
                            res.samples_seen, res.downsampling_samples_seen, res.is_training
                        )

                elif res.is_running:
                    logger.warning("Trainer server is not blocked and running, but no state is available.")

            if res.is_running:
                time.sleep(2)
            else:
                trainer_log = json.loads(res.log.value)
                break

        if training_reporter is not None:
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
