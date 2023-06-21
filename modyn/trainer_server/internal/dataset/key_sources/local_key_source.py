import os
from typing import Optional

from modyn.common.trigger_sample import TriggerSampleStorage
from modyn.trainer_server.internal.dataset.key_sources import AbstractKeySource

LOCAL_STORAGE_FOLDER = ".tmp_offline_dataset"


class LocalKeySource(AbstractKeySource):
    def __init__(self, pipeline_id: int, trigger_id: int) -> None:
        super().__init__(pipeline_id, trigger_id)

        self._trigger_sample_storage = TriggerSampleStorage(LOCAL_STORAGE_FOLDER)

    def get_keys_and_weights(self, worker_id: int, partition_id: int) -> tuple[list[int], Optional[list[float]]]:
        path = self._trigger_sample_storage._get_file_path(self._pipeline_id, self._trigger_id, partition_id, worker_id)
        file = path.parent / (path.name + ".npy")
        tuples_list = self._trigger_sample_storage._parse_file(file)

        keys = []
        weights = []
        for key, weight in tuples_list:
            keys.append(key)
            weights.append(weight)

        return keys, weights

    def get_num_data_partitions(self) -> int:
        # each file follows the structure {pipeline_id}_{trigger_id}_{partition_id}_{worker_id}

        # here we filter the files belonging to this pipeline and trigger
        this_trigger_files = list(
            filter(
                lambda file: file.startswith(f"{self._pipeline_id}_{self._trigger_id}_"),
                os.listdir(LOCAL_STORAGE_FOLDER),
            )
        )

        # then we count how many partitions we have (not just len(this_trigger_partitions) since there could be
        # multiple workers for each partition
        return len(set(file.split("_")[2] for file in this_trigger_files))

    def uses_weights(self) -> bool:
        return True

    def clean_working_directory(self) -> None:
        # remove all the files belonging to this pipeline
        if os.path.isdir(LOCAL_STORAGE_FOLDER):
            this_pipeline_files = list(
                filter(lambda file: file.startswith(f"{self._pipeline_id}_"), os.listdir(LOCAL_STORAGE_FOLDER))
            )

            for file in this_pipeline_files:
                os.remove(os.path.join(LOCAL_STORAGE_FOLDER, file))

    def clean_this_trigger_samples(self) -> None:
        # remove all the files belonging to this pipeline and trigger

        if os.path.isdir(LOCAL_STORAGE_FOLDER):
            this_trigger_files = list(
                filter(
                    lambda file: file.startswith(f"{self._pipeline_id}_{self._trigger_id}_"),
                    os.listdir(LOCAL_STORAGE_FOLDER),
                )
            )

            for file in this_trigger_files:
                os.remove(os.path.join(LOCAL_STORAGE_FOLDER, file))
