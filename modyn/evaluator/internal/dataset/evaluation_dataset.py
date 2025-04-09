import logging
from collections.abc import Callable, Generator, Iterable

import grpc
from tenacity import Retrying, after_log, before_log, retry, stop_after_attempt, wait_random_exponential
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms

from modyn.storage.internal.grpc.generated.storage_pb2 import (  # pylint: disable=no-name-in-module
    GetDataPerWorkerRequest,
    GetRequest,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.utils.utils import (
    BYTES_PARSER_FUNC_NAME,
    MAX_MESSAGE_SIZE,
    deserialize_function,
    grpc_connection_established,
    instantiate_class,
)

logger = logging.getLogger(__name__)


class EvaluationDataset(IterableDataset):
    # pylint: disable=too-many-instance-attributes, abstract-method

    def __init__(
        self,
        dataset_id: str,
        bytes_parser: str,
        serialized_transforms: list[str],
        storage_address: str,
        evaluation_id: int,
        include_labels: bool = True,
        bytes_parser_target: str | None = None,
        serialized_target_transforms: list[str] | None = None,
        tokenizer: str | None = None,
        start_timestamp: int | None = None,
        end_timestamp: int | None = None,
    ):
        self._evaluation_id = evaluation_id
        self._dataset_id = dataset_id
        self._first_call = True

        self._bytes_parser = bytes_parser
        self._serialized_transforms = serialized_transforms
        self._storage_address = storage_address
        self._transform_list: list[Callable] = []
        self._transform: Callable | None = None
        self._storagestub: StorageStub = None
        self._bytes_parser_function: Callable | None = None
        self._start_timestamp = start_timestamp
        self._end_timestamp = end_timestamp

        self._include_labels = include_labels
        self._serialized_target_transforms = serialized_target_transforms
        self._transform_target: Callable | None = None

        # Use target bytes parser if provided; otherwise, use the normal one.
        self._bytes_parser_target = bytes_parser_target if bytes_parser_target is not None else bytes_parser
        self._bytes_parser_function_target: Callable | None = None

        self._tokenizer = None
        self._tokenizer_name = tokenizer

        if tokenizer is not None:
            self._tokenizer = instantiate_class("modyn.models.tokenizers", tokenizer)

        logger.debug("Initialized EvaluationDataset.")

    def _init_transforms(self) -> None:
        self._bytes_parser_function = deserialize_function(self._bytes_parser, BYTES_PARSER_FUNC_NAME)
        self._bytes_parser_function_target = deserialize_function(self._bytes_parser_target, BYTES_PARSER_FUNC_NAME)
        self._transform = self._bytes_parser_function
        # Ensure _transform_target is always callable.
        self._transform_target = self._bytes_parser_function_target
        self._setup_composed_transform()
        self._setup_composed_target_transform()

    def _setup_composed_transform(self) -> None:
        assert self._bytes_parser_function is not None
        self._transform_list = [self._bytes_parser_function]
        for transform in self._serialized_transforms:
            function = eval(transform)  # pylint: disable=eval-used
            self._transform_list.append(function)
        if self._tokenizer is not None:
            self._transform_list.append(self._tokenizer)
        if self._transform_list:
            self._transform = transforms.Compose(self._transform_list)

    def _setup_composed_target_transform(self) -> None:
        target_transform_list = []
        if self._bytes_parser_function_target is not None:
            target_transform_list.append(self._bytes_parser_function_target)
        for transform in self._serialized_target_transforms or []:
            function = eval(transform)  # pylint: disable=eval-used
            target_transform_list.append(function)
        if self._tokenizer is not None:
            target_transform_list.append(self._tokenizer)
        if target_transform_list:
            self._transform_target = transforms.Compose(target_transform_list)

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_random_exponential(multiplier=1, min=2, max=60),
        before=before_log(logger, logging.ERROR),
        after=after_log(logger, logging.ERROR),
        reraise=True,
    )
    def _init_grpc(self) -> None:
        storage_channel = grpc.insecure_channel(
            self._storage_address,
            options=[
                ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
                ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ],
        )
        if not grpc_connection_established(storage_channel):
            raise ConnectionError(f"Could not establish gRPC connection to storage at address {self._storage_address}.")
        self._storagestub = StorageStub(storage_channel)

    def _info(self, msg: str, worker_id: int | None) -> None:
        logger.info(f"[Evaluation {self._evaluation_id}][Worker {worker_id}] {msg}")

    def _debug(self, msg: str, worker_id: int | None) -> None:
        logger.debug(f"[Evaluation {self._evaluation_id}][Worker {worker_id}] {msg}")

    @staticmethod
    def _silence_pil() -> None:
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.INFO)

    def _get_keys_from_storage(self, worker_id: int, total_workers: int) -> Iterable[list[int]]:
        self._info("Getting keys from storage", worker_id)
        processed_keys: set[int] | list[int] = []
        has_failed = False
        for attempt in Retrying(
            stop=stop_after_attempt(5),
            wait=wait_random_exponential(multiplier=1, min=2, max=60),
            reraise=True,
        ):
            with attempt:
                try:
                    req_keys = GetDataPerWorkerRequest(
                        dataset_id=self._dataset_id,
                        worker_id=worker_id,
                        total_workers=total_workers,
                        start_timestamp=self._start_timestamp,
                        end_timestamp=self._end_timestamp,
                    )
                    for resp_keys in self._storagestub.GetDataPerWorker(req_keys):
                        if not has_failed:
                            assert isinstance(processed_keys, list)
                            processed_keys.extend(resp_keys.keys)
                            yield resp_keys.keys
                        else:
                            assert isinstance(processed_keys, set)
                            new_keys = [key for key in resp_keys.keys if key not in processed_keys]
                            processed_keys.update(resp_keys.keys)
                            yield new_keys
                except grpc.RpcError as e:
                    has_failed = True
                    processed_keys = set(processed_keys) if isinstance(processed_keys, list) else processed_keys
                    self._info(
                        "gRPC error occurred, processed_keys = " + f"{processed_keys}\n{e.code()} - {e.details()}",
                        worker_id,
                    )
                    self._info(f"Stringified exception: {str(e)}", worker_id)
                    self._info(
                        f"Error occurred while asking {self._dataset_id} for worker data:\n{worker_id}", worker_id
                    )
                    self._init_grpc()
                    raise e

    def _get_data_from_storage(
        self, keys: list[int], worker_id: int | None = None
    ) -> Iterable[list[tuple[int, bytes, int, bytes | None]]]:
        processed_keys: set[int] | list[int] = []
        has_failed = False
        for attempt in Retrying(
            stop=stop_after_attempt(5),
            wait=wait_random_exponential(multiplier=1, min=2, max=60),
            reraise=True,
        ):
            with attempt:
                try:
                    request = GetRequest(dataset_id=self._dataset_id, keys=keys)
                    for response in self._storagestub.Get(request):
                        if not has_failed:
                            assert isinstance(processed_keys, list)
                            processed_keys.extend(response.keys)

                            yield list(zip(response.keys, response.samples, response.labels, response.target))
                        else:
                            assert isinstance(processed_keys, set)
                            new_keys = [key for key in response.keys if key not in processed_keys]
                            new_samples = [
                                sample
                                for key, sample in zip(response.keys, response.samples)
                                if key not in processed_keys
                            ]

                            new_labels = [
                                label for key, label in zip(response.keys, response.labels) if key not in processed_keys
                            ]
                            processed_keys.update(response.keys)

                            new_targets = [
                                target
                                for key, target in zip(response.keys, response.target)
                                if key not in processed_keys
                            ]
                            processed_keys.update(response.keys)
                            yield list(zip(new_keys, new_samples, new_labels, new_targets))
                except grpc.RpcError as e:
                    has_failed = True
                    processed_keys = set(processed_keys) if isinstance(processed_keys, list) else processed_keys
                    self._info(
                        "gRPC error occurred, processed_keys = " + f"{processed_keys}\n{e.code()} - {e.details()}",
                        worker_id,
                    )
                    self._info(f"Stringified exception: {str(e)}", worker_id)
                    self._info(f"Error occurred while asking {self._dataset_id} for keys:\n{keys}", worker_id)
                    self._init_grpc()
                    raise e

    def _get_transformed_data_tuple(self, key: int, sample: bytes, label: int | bytes | None = None) -> tuple:
        transformed_sample = self._transform(sample)  # type:ignore

        if not self._include_labels and label is not None:
            transformed_target = self._transform_target(label)  # type: ignore
            return key, transformed_sample, transformed_target

        return key, transformed_sample, label

    def __iter__(self) -> Generator:
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            total_workers = 1
        else:
            worker_id = worker_info.id
            total_workers = worker_info.num_workers

        if self._first_call:
            self._first_call = False
            self._debug("This is the first run of iter, making gRPC connections.", worker_id)
            self._init_transforms()
            self._init_grpc()
            EvaluationDataset._silence_pil()
            self._debug("gRPC initialized.", worker_id)

        assert self._transform is not None

        for keys in self._get_keys_from_storage(worker_id, total_workers):
            for data in self._get_data_from_storage(keys, worker_id):
                for key, sample, label, target in data:
                    if not self._include_labels:
                        yield self._get_transformed_data_tuple(key, sample, target)
                    yield self._get_transformed_data_tuple(key, sample, label)
