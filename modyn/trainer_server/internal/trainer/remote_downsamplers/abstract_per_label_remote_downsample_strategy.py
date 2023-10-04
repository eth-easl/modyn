from abc import abstractmethod

from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)


class AbstractPerLabelRemoteDownsamplingStrategy(AbstractRemoteDownsamplingStrategy):
    def __init__(self, pipeline_id: int, trigger_id: int, batch_size: int, params_from_selector: dict, device: str):
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, device)
        self.requires_data_label_by_label = True

    @abstractmethod
    def inform_end_of_current_label(self) -> None:
        raise NotImplementedError
