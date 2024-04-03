# pylint: disable=no-name-in-module
import json
import logging
import multiprocessing as mp
from collections import deque
from time import sleep
from typing import Any, Iterable, Optional

import grpc
from modyn.common.benchmark import Stopwatch
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    DatasetInfo,
    EvaluateModelRequest,
    EvaluateModelResponse,
    EvaluationResultRequest,
    EvaluationResultResponse,
    EvaluationStatusRequest,
    EvaluationStatusResponse,
)
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import JsonString as EvaluatorJsonString
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import MetricConfiguration
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import PythonString as EvaluatorPythonString
from modyn.evaluator.internal.grpc.generated.evaluator_pb2_grpc import EvaluatorStub
from modyn.selector.internal.grpc.generated.selector_pb2 import (
    DataInformRequest,
    DataInformResponse,
    GetNumberOfSamplesRequest,
    GetStatusBarScaleRequest,
    NumberOfSamplesResponse,
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
from modyn.supervisor.internal.evaluation_result_writer import AbstractEvaluationResultWriter
from modyn.supervisor.internal.grpc.enums import IdType, MsgType, PipelineStage
from modyn.supervisor.internal.grpc.template_msg import id_submsg, pipeline_stage_msg
from modyn.supervisor.internal.utils import EvaluationStatusReporter, TrainingStatusReporter
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
from modyn.utils import grpc_common_config, grpc_connection_established

logger = logging.getLogger(__name__)


class GRPCHandler:
    # pylint: disable=too-many-instance-attributes

    # pylint: disable=unused-argument
    def __init__(
        self,
        modyn_config: dict,
        pipeline_status_queue: Optional[mp.Queue] = None,
        training_status_queue: Optional[mp.Queue] = None,
        eval_status_queue: Optional[mp.Queue] = None,
    ) -> None:
        self.config = modyn_config
        self.pipeline_status_queue = pipeline_status_queue
        self.training_status_queue = training_status_queue
        self.eval_status_queue = eval_status_queue

        self.connected_to_storage = False
        self.connected_to_trainer_server = False
        self.connected_to_selector = False
        self.connected_to_evaluator = False

        self.storage: Optional[StorageStub] = None
        self.storage_channel: Optional[grpc.Channel] = None
        self.selector: Optional[SelectorStub] = None
        self.selector_channel: Optional[grpc.Channel] = None
        self.trainer_server: Optional[TrainerServerStub] = None
        self.trainer_server_channel: Optional[grpc.Channel] = None
        self.evaluator: Optional[EvaluatorStub] = None
        self.evaluator_channel: Optional[grpc.Channel] = None

    def init_cluster_connection(self) -> None:
        self.init_storage()
        self.init_selector()
        self.init_trainer_server()
        self.init_evaluator()

    def init_storage(self) -> None:
        assert self.config is not None
        storage_address = f"{self.config['storage']['hostname']}:{self.config['storage']['port']}"
        self.storage_channel = grpc.insecure_channel(storage_address, options=grpc_common_config())

        if not grpc_connection_established(self.storage_channel):
            raise ConnectionError(f"Could not establish gRPC connection to storage at {storage_address}.")

        self.storage = StorageStub(self.storage_channel)
        logger.info("Successfully connected to storage.")
        self.connected_to_storage = self.storage is not None

    def init_selector(self) -> None:
        assert self.config is not None
        selector_address = f"{self.config['selector']['hostname']}:{self.config['selector']['port']}"
        self.selector_channel = grpc.insecure_channel(selector_address, options=grpc_common_config())

        if not grpc_connection_established(self.selector_channel):
            raise ConnectionError(f"Could not establish gRPC connection to selector at {selector_address}.")

        self.selector = SelectorStub(self.selector_channel)
        logger.info("Successfully connected to selector.")
        self.connected_to_selector = self.selector is not None

    def init_trainer_server(self) -> None:
        assert self.config is not None
        trainer_server_address = f"{self.config['trainer_server']['hostname']}:{self.config['trainer_server']['port']}"
        self.trainer_server_channel = grpc.insecure_channel(trainer_server_address, options=grpc_common_config())

        if not grpc_connection_established(self.trainer_server_channel):
            raise ConnectionError(f"Could not establish gRPC connection to trainer server at {trainer_server_address}.")

        self.trainer_server = TrainerServerStub(self.trainer_server_channel)
        logger.info("Successfully connected to trainer server.")
        self.connected_to_trainer_server = self.trainer_server is not None

    def init_evaluator(self) -> None:
        assert self.config is not None
        evaluator_address = f"{self.config['evaluator']['hostname']}:{self.config['evaluator']['port']}"
        self.evaluator_channel = grpc.insecure_channel(evaluator_address, options=grpc_common_config())

        if not grpc_connection_established(self.evaluator_channel):
            raise ConnectionError(f"Could not establish gRPC connection to evaluator at {evaluator_address}.")

        self.evaluator = EvaluatorStub(self.evaluator_channel)
        logger.info("Successfully connected to evaluator.")
        self.connected_to_evaluator = self.evaluator is not None

    def dataset_available(self, dataset_id: str) -> bool:
        assert self.storage is not None
        assert self.connected_to_storage, "Tried to check for dataset availability, but no storage connection."
        logger.info(f"Checking whether dataset {dataset_id} is available.")

        response = self.storage.CheckAvailability(DatasetAvailableRequest(dataset_id=dataset_id))

        return response.available

    def get_new_data_since(
        self, dataset_id: str, timestamp: int
    ) -> Iterable[tuple[list[tuple[int, int, int]], dict[str, Any]]]:
        assert self.storage is not None
        if not self.connected_to_storage:
            raise ConnectionError("Tried to fetch data from storage, but no connection was made.")

        swt = Stopwatch()
        request = GetNewDataSinceRequest(dataset_id=dataset_id, timestamp=timestamp)
        response: GetNewDataSinceResponse
        swt.start("request", overwrite=True)
        for response in self.storage.GetNewDataSince(request):
            data = list(zip(response.keys, response.timestamps, response.labels))
            yield data, swt.stop()
            swt.start("request", overwrite=True)

    def get_data_in_interval(
        self, dataset_id: str, start_timestamp: int, end_timestamp: int
    ) -> Iterable[tuple[list[tuple[int, int, int]], dict[str, Any]]]:
        assert self.storage is not None
        if not self.connected_to_storage:
            raise ConnectionError("Tried to fetch data from storage, but no connection was made.")

        swt = Stopwatch()
        request = GetDataInIntervalRequest(
            dataset_id=dataset_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )
        response: GetDataInIntervalResponse
        swt.start("request", overwrite=True)
        for response in self.storage.GetDataInInterval(request):
            data = list(zip(response.keys, response.timestamps, response.labels))
            yield data, swt.stop()
            swt.start("request", overwrite=True)

    def get_time_at_storage(self) -> int:
        assert self.storage is not None
        if not self.connected_to_storage:
            raise ConnectionError("Tried to fetch data from storage, but no connection was made.")

        response: GetCurrentTimestampResponse = self.storage.GetCurrentTimestamp(
            storage_pb2.google_dot_protobuf_dot_empty__pb2.Empty()  # type: ignore
        )

        return response.timestamp

    def inform_selector(self, pipeline_id: int, data: list[tuple[int, int, int]]) -> dict[str, Any]:
        assert self.selector is not None

        keys, timestamps, labels = zip(*data)
        request = DataInformRequest(pipeline_id=pipeline_id, keys=keys, timestamps=timestamps, labels=labels)
        response: DataInformResponse = self.selector.inform_data(request)

        return json.loads(response.log.value)

    def inform_selector_and_trigger(
        self, pipeline_id: int, data: list[tuple[int, int, int]]
    ) -> tuple[int, dict[str, Any]]:
        assert self.selector is not None

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
        return trigger_id, json.loads(response.log.value)

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
        self, pipeline_id: int, trigger_id: int, pipeline_config: dict, previous_model_id: Optional[int]
    ) -> int:
        assert self.trainer_server is not None
        if not self.connected_to_trainer_server:
            raise ConnectionError("Tried to start training at trainer server, but not there is no gRPC connection.")

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

        if "num_prefetched_partitions" in pipeline_config["training"]:
            num_prefetched_partitions = pipeline_config["training"]["num_prefetched_partitions"]
        else:
            if "prefetched_partitions" in pipeline_config["training"]:
                raise ValueError(
                    "Found `prefetched_partitions` instead of `num_prefetched_partitions`in training configuration."
                    + " Please rename/remove that configuration"
                )
            logger.warning("Number of prefetched partitions not explicitly given in training config - defaulting to 1.")
            num_prefetched_partitions = 1

        if "parallel_prefetch_requests" in pipeline_config["training"]:
            parallel_prefetch_requests = pipeline_config["training"]["parallel_prefetch_requests"]
        else:
            logger.warning(
                "Number of parallel prefetch requests not explicitly given in training config - defaulting to 1."
            )
            parallel_prefetch_requests = 1

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

        if "grad_scaler_config" in pipeline_config["training"]:
            grad_scaler_config = pipeline_config["training"]["grad_scaler_config"]
        else:
            grad_scaler_config = {}

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
        assert self.selector is not None

        request = GetNumberOfSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id)
        response: NumberOfSamplesResponse = self.selector.get_number_of_samples(request)

        return response.num_samples

    def get_status_bar_scale(self, pipeline_id: int) -> int:
        assert self.selector is not None

        request = GetStatusBarScaleRequest(pipeline_id=pipeline_id)
        response: StatusBarScaleResponse = self.selector.get_status_bar_scale(request)

        return response.status_bar_scale

    # pylint: disable=too-many-nested-blocks
    def wait_for_training_completion(
        self, training_id: int, pipeline_id: int, trigger_id: int
    ) -> dict[str, Any]:  # pragma: no cover
        assert self.training_status_queue is not None
        assert self.pipeline_status_queue is not None
        assert self.trainer_server is not None
        if not self.connected_to_trainer_server:
            raise ConnectionError(
                "Tried to wait for training to finish at trainer server, but not there is no gRPC connection."
            )
        logger.debug("wait for training completion")

        total_samples = self.get_number_of_samples(pipeline_id, trigger_id)
        status_bar_scale = self.get_status_bar_scale(pipeline_id)
        training_reporter = TrainingStatusReporter(
            self.training_status_queue, trigger_id, training_id, total_samples, status_bar_scale
        )
        training_reporter.create_tracker()
        self.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.WAIT_FOR_TRAINING_COMPLETION, MsgType.ID, id_submsg(IdType.TRAINING, training_id), True
            )
        )

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
                    # status_tracker.progress_counter(res.samples_seen, res.downsampling_samples_seen, res.is_training)

                elif res.is_running:
                    logger.warning("Trainer server is not blocked and running, but no state is available.")

            if res.is_running:
                sleep(2)
            else:
                trainer_log = json.loads(res.log.value)
                break

        training_reporter.close_counter()
        # status_tracker.close_counter()
        self.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.TRAINING_COMPLETED, MsgType.ID, id_submsg(IdType.TRAINING, training_id), True
            )
        )
        logger.debug("Training completed")

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

    def seed_selector(self, seed: int) -> None:
        assert self.selector is not None

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

    def start_evaluation(self, model_id: int, pipeline_config: dict) -> dict[int, EvaluationStatusReporter]:
        assert self.evaluator is not None
        assert self.eval_status_queue is not None
        if not self.connected_to_evaluator:
            raise ConnectionError("Tried to start evaluation at evaluator, but there is no gRPC connection.")
        device = pipeline_config["evaluation"]["device"]

        evaluations: dict[int, EvaluationStatusReporter] = {}

        for dataset in pipeline_config["evaluation"]["datasets"]:
            dataset_id = dataset["dataset_id"]
            req = GRPCHandler._prepare_evaluation_request(dataset, model_id, device)
            response: EvaluateModelResponse = self.evaluator.evaluate_model(req)

            if not response.evaluation_started:
                logger.error(f"Starting evaluation for dataset {dataset_id} did go wrong: {response}.")
            else:
                evaluation_id = response.evaluation_id
                logger.info(f"Started evaluation {evaluation_id} on dataset {dataset_id}.")
                evaluations[evaluation_id] = EvaluationStatusReporter(
                    self.eval_status_queue, evaluation_id, dataset_id, response.dataset_size
                )
                evaluations[evaluation_id].create_tracker()

        return evaluations

    @staticmethod
    def _prepare_evaluation_request(
        dataset_config: dict, model_id: int, device: str, start_timestamp: int = 0, end_timestamp: int = 0
    ) -> EvaluateModelRequest:
        dataset_id = dataset_config["dataset_id"]

        if "transformations" in dataset_config:
            transform_list = dataset_config["transformations"]
        else:
            transform_list = []

        if "label_transformer_function" in dataset_config:
            label_transformer = dataset_config["label_transformer_function"]
        else:
            label_transformer = ""

        bytes_parser_function = dataset_config["bytes_parser_function"]
        batch_size = dataset_config["batch_size"]
        dataloader_workers = dataset_config["dataloader_workers"]
        metrics = []
        for metric in dataset_config["metrics"]:
            name = metric["name"]
            if "config" in metric:
                metric_config = json.dumps(metric["config"])
            else:
                metric_config = "{}"

            if "evaluation_transformer_function" in metric:
                evaluation_transformer = metric["evaluation_transformer_function"]
            else:
                evaluation_transformer = ""

            metrics.append(
                MetricConfiguration(
                    name=name,
                    config=EvaluatorJsonString(value=metric_config),
                    evaluation_transformer=EvaluatorPythonString(value=evaluation_transformer),
                )
            )

        start_evaluation_kwargs = {
            "model_id": model_id,
            "dataset_info": DatasetInfo(
                dataset_id=dataset_id,
                num_dataloaders=dataloader_workers,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            ),
            "device": device,
            "batch_size": batch_size,
            "metrics": metrics,
            "transform_list": transform_list,
            "bytes_parser": EvaluatorPythonString(value=bytes_parser_function),
            "label_transformer": EvaluatorPythonString(value=label_transformer),
        }

        return EvaluateModelRequest(**start_evaluation_kwargs)

    def wait_for_evaluation_completion(
        self, training_id: int, evaluations: dict[int, EvaluationStatusReporter]
    ) -> None:
        assert self.pipeline_status_queue is not None
        assert self.evaluator is not None
        if not self.connected_to_evaluator:
            raise ConnectionError("Tried to wait for evaluation to finish, but not there is no gRPC connection.")

        logger.debug("wait for evaluation completion")
        self.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.WAIT_FOR_EVALUATION_COMPLETION, MsgType.ID, id_submsg(IdType.TRAINING, training_id), True
            )
        )

        # We are using a deque here in order to fetch the status of each evaluation
        # sequentially in a round-robin manner.
        working_queue: deque[int] = deque()
        blocked_in_a_row: dict[int, int] = {}
        for evaluation_id, evaluation_reporter in evaluations.items():
            evaluation_reporter.create_counter(training_id)
            working_queue.append(evaluation_id)
            blocked_in_a_row[evaluation_id] = 0

        while working_queue:
            current_evaluation_id = working_queue.popleft()
            current_evaluation_reporter = evaluations[current_evaluation_id]
            req = EvaluationStatusRequest(evaluation_id=current_evaluation_id)
            res: EvaluationStatusResponse = self.evaluator.get_evaluation_status(req)

            if not res.valid:
                exception_msg = f"Evaluation {current_evaluation_id} is invalid at server:\n{res}\n"
                logger.warning(exception_msg)
                current_evaluation_reporter.end_counter(error=True, exception_msg=exception_msg)
                continue

            if res.blocked:
                blocked_in_a_row[current_evaluation_id] += 1
                if blocked_in_a_row[current_evaluation_id] >= 3:
                    logger.warning(
                        f"Evaluator returned {blocked_in_a_row} blocked responses in a row, cannot update status."
                    )
            else:
                blocked_in_a_row[current_evaluation_id] = 0

                if res.HasField("exception") and res.exception is not None:
                    exception_msg = f"Exception at evaluator occurred:\n{res.exception}\n\n"
                    logger.warning(exception_msg)
                    current_evaluation_reporter.end_counter(error=True, exception_msg=exception_msg)
                    continue
                if not res.is_running:
                    current_evaluation_reporter.end_counter(error=False)
                    continue
                if res.state_available:
                    assert res.HasField("samples_seen") and res.HasField(
                        "batches_seen"
                    ), f"Inconsistent server response:\n{res}"

                    current_evaluation_reporter.progress_counter(res.samples_seen)
                elif res.is_running:
                    logger.warning("Evaluator is not blocked and is running, but no state is available.")

            working_queue.append(current_evaluation_id)
            sleep(1)

        self.pipeline_status_queue.put(
            pipeline_stage_msg(
                PipelineStage.EVALUATION_COMPLETED, MsgType.ID, id_submsg(IdType.TRAINING, training_id), True
            )
        )
        logger.debug("Evaluation completed")

    def store_evaluation_results(
        self,
        evaluation_result_writers: list[AbstractEvaluationResultWriter],
        evaluations: dict[int, EvaluationStatusReporter],
    ) -> None:
        assert self.evaluator is not None
        if not self.connected_to_evaluator:
            raise ConnectionError("Tried to wait for evaluation to finish, but not there is no gRPC connection.")

        for evaluation_id in evaluations:
            req = EvaluationResultRequest(evaluation_id=evaluation_id)
            res: EvaluationResultResponse = self.evaluator.get_evaluation_result(req)

            if not res.valid:
                logger.warning(f"Cannot get the evaluation result for evaluation {evaluation_id}")
                continue
            dataset_id = evaluations[evaluation_id].dataset_id
            dataset_size = evaluations[evaluation_id].dataset_size

            for result_writer in evaluation_result_writers:
                result_writer.add_evaluation_data(dataset_id, dataset_size, res.evaluation_data)

        for result_writer in evaluation_result_writers:
            result_writer.store_results()
