from typing import Optional

from modyn.trainer_server.internal.dataset.key_sources import AbstractKeySource
from modyn.trainer_server.internal.dataset.local_dataset_reader import LocalDatasetReader


class LocalKeySource(AbstractKeySource):
    def __init__(self, pipeline_id: int, trigger_id: int, number_of_workers: int) -> None:
        super().__init__(pipeline_id, trigger_id)

        self._local_dataset_reader = LocalDatasetReader(pipeline_id, trigger_id, number_of_workers)

    def get_keys_and_weights(self, worker_id: int, partition_id: int) -> tuple[list[int], Optional[list[float]]]:
        keys, weights = self._local_dataset_reader.get_keys_and_weights(partition_id, worker_id)

        return keys, weights

    def get_num_data_partitions(self) -> int:
        return self._local_dataset_reader.get_number_of_partitions()

    def uses_weights(self) -> bool:
        return True
