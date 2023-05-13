import numpy as np
from modyn.common.trigger_sample.trigger_sample_storage import TriggerSampleStorage


class LocalDatasetHandler(TriggerSampleStorage):
    def __init__(self, pipeline_id: int, number_of_samples_per_file: int = -1) -> None:
        super().__init__(".tmp_offline_dataset")
        self.current_file_index = 0
        self.current_sample_index = 0
        self.file_size = number_of_samples_per_file
        self.pipeline_id = pipeline_id
        if self.file_size > 0:
            self.output_samples_list = np.empty(self.file_size, dtype=np.dtype("i8,f8"))

    def inform_samples(self, input_samples_list: np.ndarray) -> None:
        assert self.file_size > 0
        assert self.output_samples_list is not None
        for element in input_samples_list:
            self.output_samples_list[self.current_sample_index] = element
            self.current_sample_index += 1

            if self.current_sample_index == self.file_size:
                self.samples_ready()

    def samples_ready(self) -> None:
        assert self.file_size > 0
        assert self.output_samples_list is not None
        self.save_trigger_sample(self.pipeline_id, 0, self.current_file_index, self.output_samples_list, -1)
        self.current_sample_index = 0
        self.output_samples_list = np.empty(self.file_size, dtype=np.dtype("i8,f8"))
        self.current_file_index += 1

    def get_keys_and_weights(
        self, worker_id: int, partition_id: int  # pylint: disable=unused-argument
    ) -> tuple[list[int], list[float]]:
        tuples_list = self.get_trigger_samples(self.pipeline_id, 0, partition_id)
        keys = []
        weights = []
        for key, weight in tuples_list:
            keys.append(key)
            weights.append(weight)
        return keys, weights
