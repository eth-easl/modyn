# pylint: disable=no-name-in-module
from __future__ import annotations

import json
import logging
from collections import deque
from time import sleep
from typing import Any, Iterable, Optional, Sequence

import grpc
from modyn.common.benchmark import Stopwatch
from modyn.common.grpc.grpc_helpers import TrainerServerGRPCHandlerMixin
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    DatasetInfo,
    EvaluateModelRequest,
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
from modyn.supervisor.internal.utils import EvaluationStatusReporter
from modyn.utils import grpc_common_config, grpc_connection_established

logger = logging.getLogger(__name__)


class GRPCHandler(TrainerServerGRPCHandlerMixin):

    # pylint: disable=unused-argument
    def __init__(self, modyn_config: dict) -> None:
        super().__init__(modyn_config=modyn_config)
        self.config = modyn_config

        self.connected_to_storage = False
        self.connected_to_selector = False
        self.connected_to_evaluator = False

        self.storage: Optional[StorageStub] = None
        self.storage_channel: Optional[grpc.Channel] = None
        self.selector: Optional[SelectorStub] = None
        self.selector_channel: Optional[grpc.Channel] = None
        self.evaluator: Optional[EvaluatorStub] = None
        self.evaluator_channel: Optional[grpc.Channel] = None

    def init_cluster_connection(self) -> None:
        self.init_storage()
        self.init_selector()
        self.init_trainer_server()
        self.init_evaluator()

    def init_storage(self) -> None:
        storage_address = f"{self.config['storage']['hostname']}:{self.config['storage']['port']}"
        self.storage_channel = grpc.insecure_channel(storage_address, options=grpc_common_config())

        if not grpc_connection_established(self.storage_channel):
            raise ConnectionError(f"Could not establish gRPC connection to storage at {storage_address}.")

        self.storage = StorageStub(self.storage_channel)
        logger.info("Successfully connected to storage.")
        self.connected_to_storage = self.storage is not None

    def init_selector(self) -> None:
        selector_address = f"{self.config['selector']['hostname']}:{self.config['selector']['port']}"
        self.selector_channel = grpc.insecure_channel(selector_address, options=grpc_common_config())

        if not grpc_connection_established(self.selector_channel):
            raise ConnectionError(f"Could not establish gRPC connection to selector at {selector_address}.")

        self.selector = SelectorStub(self.selector_channel)
        logger.info("Successfully connected to selector.")
        self.connected_to_selector = self.selector is not None

    def init_evaluator(self) -> None:
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

    def get_new_data_since(self, dataset_id: str, timestamp: int) -> Iterable[tuple[list[tuple[int, int, int]], int]]:
        """Returns:
        tuple containing actual data and fetch time in milliseconds
        """
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
    ) -> Iterable[tuple[list[tuple[int, int, int]], int]]:
        """Returns:
        tuple containing actual data and fetch time in milliseconds
        """
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

    # pylint: disable=too-many-locals
    @staticmethod
    def prepare_evaluation_request(
        dataset_config: dict,
        model_id: int,
        device: str,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
    ) -> EvaluateModelRequest:
        dataset_id = dataset_config["dataset_id"]
        transform_list = dataset_config.get("transformations") or []
        label_transformer = dataset_config.get("label_transformer_function") or ""

        bytes_parser_function = dataset_config["bytes_parser_function"]
        batch_size = dataset_config["batch_size"]
        dataloader_workers = dataset_config["dataloader_workers"]
        metrics = []
        for metric in dataset_config["metrics"]:
            name = metric["name"]
            metric_config = json.dumps(metric.get("config") or {})
            evaluation_transformer = metric.get("evaluation_transformer_function") or ""

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

        if dataset_config.get("tokenizer"):
            tokenizer = dataset_config["tokenizer"]
            start_evaluation_kwargs["tokenizer"] = EvaluatorPythonString(value=tokenizer)

        return EvaluateModelRequest(**start_evaluation_kwargs)

    def wait_for_evaluation_completion(
        self, training_id: int, evaluations: dict[int, EvaluationStatusReporter]
    ) -> None:
        assert self.evaluator is not None
        if not self.connected_to_evaluator:
            raise ConnectionError("Tried to wait for evaluation to finish, but not there is no gRPC connection.")

        logger.debug("wait for evaluation completion")

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

        logger.debug("Evaluation completed")

    def store_evaluation_results(
        self,
        evaluation_result_writers: Sequence[AbstractEvaluationResultWriter],
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
