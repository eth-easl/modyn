from typing import Optional

from modyn.common.trigger_sample import TriggerSampleStorage
from modyn.trainer_server.internal.dataset.key_sources import AbstractKeySource


class LocalKeySource(AbstractKeySource):
    def __init__(self, pipeline_id: int, trigger_id: int, offline_dataset_path: str) -> None:
        super().__init__(pipeline_id, trigger_id)

        self._trigger_sample_storage = TriggerSampleStorage(offline_dataset_path)
        self.offline_dataset_path = offline_dataset_path

    def get_keys_and_weights(self, worker_id: int, partition_id: int) -> tuple[list[int], Optional[list[float]]]:
        path = self._trigger_sample_storage.get_file_path(self._pipeline_id, self._trigger_id, partition_id, worker_id)
        tuples_list = self._trigger_sample_storage.parse_file(path)

        if len(tuples_list) == 0:
            return [], []
        keys, weights = zip(*tuples_list)  # type: ignore

        return list(keys), list(weights)

    def uses_weights(self) -> bool:
        return True

    def init_worker(self) -> None:
        pass

    def get_num_data_partitions(self) -> int:
        return self._trigger_sample_storage.get_trigger_num_data_partitions(self._pipeline_id, self._trigger_id)

    def end_of_trigger_cleaning(self) -> None:
        self._trigger_sample_storage.clean_trigger_data(self._pipeline_id, self._trigger_id)

    def get_number_of_samples(self) -> int:
        return self._trigger_sample_storage.get_number_of_samples(self._pipeline_id, self._trigger_id)
