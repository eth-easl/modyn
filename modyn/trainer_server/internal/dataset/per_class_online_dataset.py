import gc
import logging
from typing import Generator

from modyn.trainer_server.internal.dataset.online_dataset import OnlineDataset
from torch.utils.data import get_worker_info

logger = logging.getLogger(__name__)


class PerClassOnlineDataset(OnlineDataset):
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
    ):
        super().__init__(
            pipeline_id,
            trigger_id,
            dataset_id,
            bytes_parser,
            serialized_transforms,
            storage_address,
            selector_address,
            training_id,
        )

        self.filtered_label = None

    # pylint: disable=too-many-locals, too-many-branches
    def __iter__(self) -> Generator:
        assert self.filtered_label is not None
        worker_info = get_worker_info()
        if worker_info is None:
            # Non-multithreaded data loading. We use worker_id 0.
            worker_id = 0
        else:
            worker_id = worker_info.id

        if self._first_call:
            self._first_call = False
            self._debug("This is the first run of iter, making gRPC connections.", worker_id)
            # We have to initialize transformations and gRPC connections here to do it per dataloader worker,
            # otherwise the transformations/gRPC connections cannot be pickled for the new processes.
            self._init_transforms()
            self._init_grpc()
            self._key_source.init_worker()
            self._uses_weights = self._key_source.uses_weights()
            self._silence_pil()
            self._debug("gRPC initialized.", worker_id)

        assert self._transform is not None
        self._num_partitions = self._key_source.get_num_data_partitions()
        self._info(f"Total number of partitions will be {self._num_partitions}", worker_id)

        keys, data, labels, weights = self._get_data(worker_id=worker_id, partition_id=0)

        for partition in range(self._num_partitions):
            num_samples_on_this_partition = len(keys)
            # We (arbitrarily) fetch the next partition when we have seen 80% of the current partition
            fetch_next_partition_idx = int(num_samples_on_this_partition * 0.8)
            self._info(f"Train on partition {partition}, on {num_samples_on_this_partition} batches", worker_id)

            for idx, data_tuple in self._get_data_iterator(keys, data, labels, weights):
                key, sample, label, weight = self._unpack_data_tuple(data_tuple)

                if partition < self._num_partitions - 1 and idx == fetch_next_partition_idx:
                    # TODO(#175) in case this blocks training
                    new_keys, new_data, new_labels, new_weights = self._get_data(
                        worker_id=worker_id, partition_id=partition + 1
                    )
                if label == self.filtered_label:
                    yield self._yield_samples(key, sample, label, weight)

            # this should mean we keep only two partitions in mem
            if partition < self._num_partitions - 1:
                del keys
                del data
                del labels
                del weights
                keys, data, labels, weights = new_keys, new_data, new_labels, new_weights
                del new_keys
                del new_data
                del new_labels
                del new_weights
                gc.collect()
