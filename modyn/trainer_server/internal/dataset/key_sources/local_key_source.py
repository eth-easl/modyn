import os
from typing import Optional

from modyn.common.trigger_sample import TriggerSampleStorage
from modyn.trainer_server.internal.dataset.key_sources import AbstractKeySource


class LocalKeySource(AbstractKeySource):
    def __init__(self, pipeline_id: int, trigger_id: int, offline_dataset_path: str) -> None:
        super().__init__(pipeline_id, trigger_id)

        self._trigger_sample_storage = TriggerSampleStorage(offline_dataset_path)
        self.offline_dataset_path = offline_dataset_path

    def get_keys_and_weights(self, worker_id: int, partition_id: int) -> tuple[list[int], Optional[list[float]]]:
        path = self._trigger_sample_storage._get_file_path(self._pipeline_id, self._trigger_id, partition_id, worker_id)
        file = path.parent / (path.name + ".npy")
        tuples_list = self._trigger_sample_storage._parse_file(file)

        keys, weights = zip(*tuples_list)

        return list(keys), list(weights)

    def get_num_data_partitions(self) -> int:
        # each file follows the structure {pipeline_id}_{trigger_id}_{partition_id}_{worker_id}

        # here we filter the files belonging to this pipeline and trigger
        this_trigger_files = list(
            filter(
                lambda file: file.startswith(f"{self._pipeline_id}_{self._trigger_id}_"),
                os.listdir(self.offline_dataset_path),
            )
        )

        # then we count how many partitions we have (not just len(this_trigger_partitions) since there could be
        # multiple workers for each partition
        return len(set(file.split("_")[2] for file in this_trigger_files))

    def uses_weights(self) -> bool:
        return True

    def init_worker(self) -> None:
        pass

    def end_of_trigger_cleaning(self) -> None:
        # remove all the files belonging to this pipeline and trigger

        if os.path.isdir(self.offline_dataset_path):
            this_trigger_files = list(
                filter(
                    lambda file: file.startswith(f"{self._pipeline_id}_{self._trigger_id}_"),
                    os.listdir(self.offline_dataset_path),
                )
            )

            for file in this_trigger_files:
                os.remove(os.path.join(self.offline_dataset_path, file))
