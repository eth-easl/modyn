import numpy as np
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsample_strategy import (
    AbstractRemoteDownsamplingStrategy,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.sample_then_batch_temporary_storage import (
    SampleThenBatchTemporaryStorage,
)


class AbstractRemoteDownsamplingTemporaryStorage(AbstractRemoteDownsamplingStrategy):
    def __init__(self, pipeline_id: int, trigger_id: int, batch_size: int, params_from_selector: dict) -> None:
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector)

        if self.sample_then_batch:
            assert "maximum_keys_in_memory" in params_from_selector
            self.sample_then_batch_temporary_storage = SampleThenBatchTemporaryStorage(
                self.pipeline_id,
                self.trigger_id,
                self.batch_size,
                self.downsampled_batch_ratio,
                params_from_selector["maximum_keys_in_memory"],
            )
            self._next_file_idx = 0
            self._total_files = 0

    def setup_sample_then_batch(self) -> None:
        assert self.sample_then_batch
        self._sampling_concluded = False
        self.sample_then_batch_temporary_storage.reset_temporary_storage()

    def end_sample_then_batch(self) -> None:
        assert self.sample_then_batch
        self._sampling_concluded = True
        self.sample_then_batch_temporary_storage.end_accumulation()
        self._next_file_idx = 0
        self._total_files = self.sample_then_batch_temporary_storage.get_total_number_of_files()

    def samples_available(self) -> bool:
        assert self._sampling_concluded
        assert self.sample_then_batch
        return self._next_file_idx < self._total_files

    def get_samples(self) -> np.ndarray:
        assert self.samples_available()
        samples = self.sample_then_batch_temporary_storage.get_samples_per_file(self._next_file_idx)
        self._next_file_idx += 1
        return samples
