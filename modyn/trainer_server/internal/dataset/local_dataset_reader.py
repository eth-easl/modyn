import os

from modyn.common.trigger_sample.trigger_sample_storage import TriggerSampleStorage

LOCAL_STORAGE_FOLDER = ".tmp_offline_dataset"


class LocalDatasetReader(TriggerSampleStorage):
    """
    Class that wraps TriggerSampleStorage to use it as a local storage for samples.

    This class is used to read samples (stored with LocalDatasetWriter) and supply them in the usual format (list
    of keys, list of weights)
    """

    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        number_of_workers: int,
    ) -> None:
        super().__init__(LOCAL_STORAGE_FOLDER)
        # files are numbered from 0. Each file has a size of number_of_samples_per_file

        self.pipeline_id = pipeline_id
        self.trigger_id = trigger_id
        self.number_of_workers = number_of_workers

    def get_keys_and_weights(
        self,
        partition_id: int,
        worker_id: int,  # pylint: disable=unused-argument
    ) -> tuple[list[int], list[float]]:
        path = self._get_file_name(self.pipeline_id, self.trigger_id, partition_id, worker_id)
        file = path.parent / (path.name + ".npy")
        tuples_list = self._parse_file(file)

        keys = []
        weights = []
        for key, weight in tuples_list:
            keys.append(key)
            weights.append(weight)
        return keys, weights

    def clean_working_directory(self) -> None:
        # remove all the files belonging to this pipeline
        if os.path.isdir(LOCAL_STORAGE_FOLDER):
            this_pipeline_files = list(
                filter(lambda file: file.startswith(f"{self.pipeline_id}_"), os.listdir(LOCAL_STORAGE_FOLDER))
            )

            for file in this_pipeline_files:
                os.remove(os.path.join(LOCAL_STORAGE_FOLDER, file))

    def clean_this_trigger_samples(self) -> None:
        # remove all the files belonging to this pipeline and trigger

        if os.path.isdir(LOCAL_STORAGE_FOLDER):
            this_trigger_files = list(
                filter(
                    lambda file: file.startswith(f"{self.pipeline_id}_{self.trigger_id}_"),
                    os.listdir(LOCAL_STORAGE_FOLDER),
                )
            )

            for file in this_trigger_files:
                os.remove(os.path.join(LOCAL_STORAGE_FOLDER, file))

    def get_number_of_partitions(self) -> int:
        # each file follows the structure {pipeline_id}_{trigger_id}_{partition_id}_{worker_id}

        # here we filter the files belonging to this pipeline and trigger
        this_trigger_files = list(
            filter(
                lambda file: file.startswith(f"{self.pipeline_id}_{self.trigger_id}_"), os.listdir(LOCAL_STORAGE_FOLDER)
            )
        )

        # then we count how many partitions we have (not just len(this_trigger_partitions) since there could be
        # multiple workers for each partition
        return len(set(file.split("_")[2] for file in this_trigger_files))
