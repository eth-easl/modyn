from abc import abstractmethod

from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)


class AbstractPerLabelRemoteDownsamplingStrategy(AbstractRemoteDownsamplingStrategy):
    @abstractmethod
    def inform_end_of_current_label(self) -> None:
        raise NotImplementedError
