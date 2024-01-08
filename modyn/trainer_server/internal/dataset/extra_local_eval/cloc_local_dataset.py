# pylint: skip-file

import io
import json
import logging
import os
import pathlib
import threading
from typing import Any, Callable, Generator, Iterator, Optional, Tuple

from modyn.common.benchmark.stopwatch import Stopwatch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms

logger = logging.getLogger(__name__)


class ClocLocalDataset(IterableDataset):
    # pylint: disable=too-many-instance-attributes, abstract-method

    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        dataset_id: str,
        bytes_parser: str,
        serialized_transforms: list[str],
        storage_address: str,
        selector_address: str,
        training_id: int,
        num_prefetched_partitions: int,
        parallel_prefetch_requests: int,
        tokenizer: Optional[str],
        log_path: Optional[pathlib.Path],
    ):
        self._pipeline_id = pipeline_id
        self._trigger_id = trigger_id
        self._training_id = training_id
        self._dataset_id = dataset_id
        self._first_call = True
        self._num_prefetched_partitions = num_prefetched_partitions
        self._parallel_prefetch_requests = parallel_prefetch_requests

        self._bytes_parser = bytes_parser
        self._serialized_transforms = serialized_transforms
        self._storage_address = storage_address
        self._selector_address = selector_address
        self._transform_list: list[Callable] = []
        self._transform: Optional[Callable] = None
        self._log_path = log_path
        self._log: dict[str, Any] = {"partitions": {}}
        self._log_lock: Optional[threading.Lock] = None
        self._sw = Stopwatch()
        self._cloc_path = "/tmp/cloc"

        if log_path is None:
            logger.warning("Did not provide log path for ClocDataset - logging disabled.")

        logger.debug("Initialized ClocDataset.")

    @staticmethod
    def bytes_parser_function(data: memoryview) -> Image:
        return Image.open(io.BytesIO(data)).convert("RGB")

    def _setup_composed_transform(self) -> None:
        self._transform_list = [
            ClocLocalDataset.bytes_parser_function,
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self._transform = transforms.Compose(self._transform_list)

    def _init_transforms(self) -> None:
        self._setup_composed_transform()

    def _silence_pil(self) -> None:  # pragma: no cover
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.INFO)  # by default, PIL on DEBUG spams the console

    def _info(self, msg: str, worker_id: Optional[int]) -> None:  # pragma: no cover
        logger.info(f"[Training {self._training_id}][PL {self._pipeline_id}][Worker {worker_id}] {msg}")

    def _debug(self, msg: str, worker_id: Optional[int]) -> None:  # pragma: no cover
        logger.debug(f"[Training {self._training_id}][PL {self._pipeline_id}][Worker {worker_id}] {msg}")

    def _get_transformed_data_tuple(
        self, key: int, sample: memoryview, label: int, weight: Optional[float]
    ) -> Optional[Tuple]:
        self._sw.start("transform", resume=True)
        # mypy complains here because _transform has unknown type, which is ok
        tranformed_sample = self._transform(sample)  # type: ignore
        self._sw.stop("transform")
        return key, tranformed_sample, label

    def _persist_log(self, worker_id: int) -> None:
        if self._log_path is None:
            return

        assert self._log_lock is not None

        with self._log_lock:
            if "PYTEST_CURRENT_TEST" in os.environ:
                json.dumps(self._log)  # Enforce serialization to catch issues
                return  # But don't actually store in tests

            log_file = f"{self._log_path / str(worker_id)}.log"
            self._log["transform"] = self._sw.measurements.get("transform", 0)
            self._log["wait_for_later_partitions"] = self._sw.measurements.get("wait_for_later_partitions", 0)
            self._log["wait_for_initial_partition"] = self._sw.measurements.get("wait_for_initial_partition", 0)

            with open(log_file, "w", encoding="utf-8") as logfile:
                json.dump(self._log, logfile)

    def cloc_generator(
        self, worker_id: int, num_workers: int
    ) -> Iterator[tuple[int, memoryview, int, Optional[float]]]:
        self._info("Globbing paths", worker_id)

        pathlist = sorted(pathlib.Path(self._cloc_path).glob("*.jpg"))
        self._info("Paths globbed", worker_id)

        def split(a, n):
            k, m = divmod(len(a), n)
            return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

        pathgen = split(pathlist, num_workers)
        worker_paths = next(x for i, x in enumerate(pathgen) if i == worker_id)
        self._info(f"Got {len(worker_paths)} paths.", worker_id)

        sample_idx = 0
        for path in worker_paths:
            path = pathlib.Path(path)
            label_path = path.with_suffix(".label")

            with open(path, "rb") as file:
                data = file.read()
            with open(label_path, "rb") as file:
                label = int(file.read().decode("utf-8"))

            yield sample_idx, memoryview(data), label, None
            sample_idx = sample_idx + 1

    def __iter__(self) -> Generator:
        worker_info = get_worker_info()
        if worker_info is None:
            # Non-multithreaded data loading. We use worker_id 0.
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        if self._first_call:
            self._first_call = False
            self._debug("This is the first run of iter, making gRPC connections.", worker_id)
            # We have to initialize transformations and gRPC connections here to do it per dataloader worker,
            # otherwise the transformations/gRPC connections cannot be pickled for the new processes.
            self._init_transforms()
            self._uses_weights = False
            self._silence_pil()
            self._sw = Stopwatch()
            self._log_lock = threading.Lock()

        assert self._transform is not None

        for data_tuple in self.cloc_generator(worker_id, num_workers):
            if (transformed_tuple := self._get_transformed_data_tuple(*data_tuple)) is not None:
                yield transformed_tuple

        self._persist_log(worker_id)

    def end_of_trigger_cleaning(self) -> None:
        pass
