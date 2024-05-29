from typing import List, Optional, Tuple

from modyn.config import CoresetSelectionStrategy, DownsamplingConfig, MultiDownsamplingConfig
from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.selector.internal.selector_strategies.downsampling_strategies.utils import instantiate_downsampler
from modyn.selector.internal.storage_backend import AbstractStorageBackend


class DownsamplingScheduler:
    def __init__(
        self,
        modyn_config: dict,
        pipeline_id: int,
        downsampling_configs: List[DownsamplingConfig],
        downsampling_thresholds: list[int],
        maximum_keys_in_memory: int,
    ):
        self.modyn_config = modyn_config
        self.pipeline_id = pipeline_id
        self.dowsampling_configs = downsampling_configs
        self.dowsampling_thresholds: list[int] = downsampling_thresholds
        self.downsampler_index = 0
        # transition from downsampler[i] to downsampler[i+1] happens before the thresholds[i]-th trigger

        self._validate_threshold()

        self.maximum_keys_in_memory = maximum_keys_in_memory

        self.current_downsampler, self.next_threshold = self._get_next_downsampler_and_threshold()

    def _validate_threshold(self) -> None:
        if not len(self.dowsampling_configs) == len(self.dowsampling_thresholds) + 1:
            raise ValueError("You must specify a threshold for each transition")
        if not sorted(self.dowsampling_thresholds) == self.dowsampling_thresholds:
            raise ValueError("Thresholds must be monotonically increasing")
        if not len(self.dowsampling_thresholds) == len(set(self.dowsampling_thresholds)):
            raise ValueError("Thresholds must be unique")

    def _get_next_downsampler_and_threshold(self) -> Tuple[AbstractDownsamplingStrategy, Optional[int]]:
        next_downsampler = instantiate_downsampler(
            self.dowsampling_configs[self.downsampler_index],
            self.modyn_config,
            self.pipeline_id,
            self.maximum_keys_in_memory,
        )
        # after instantiating the last downsampler, we simply set to none the next threshold
        next_threshold = (
            self.dowsampling_thresholds[self.downsampler_index]
            if self.downsampler_index < len(self.dowsampling_thresholds)
            else None
        )
        return next_downsampler, next_threshold

    @property
    def requires_remote_computation(self) -> bool:
        return self.current_downsampler.requires_remote_computation

    @property
    def downsampling_strategy(self) -> str:
        assert hasattr(
            self.current_downsampler, "remote_downsampling_strategy_name"
        ), "Your downsampler must specify the remote_downsampling_strategy_name"
        return self.current_downsampler.remote_downsampling_strategy_name

    @property
    def downsampling_params(self) -> dict:
        return self.current_downsampler.downsampling_params

    @property
    def training_status_bar_scale(self) -> int:
        return self.current_downsampler.status_bar_scale

    def inform_next_trigger(self, next_trigger_id: int, selector_storage_backend: AbstractStorageBackend) -> None:
        if self.next_threshold is not None and next_trigger_id >= self.next_threshold:
            self.downsampler_index += 1
            self.current_downsampler, self.next_threshold = self._get_next_downsampler_and_threshold()

        self.current_downsampler.inform_next_trigger(next_trigger_id, selector_storage_backend)


def instantiate_scheduler(
    config: CoresetSelectionStrategy, modyn_config: dict, pipeline_id: int
) -> DownsamplingScheduler:
    if isinstance(config.downsampling_config, MultiDownsamplingConfig):
        # real scheduler
        list_of_downsamplers = config.downsampling_config.downsampling_list
        list_of_thresholds = config.downsampling_config.downsampling_thresholds
    else:
        # just use one strategy, so fake scheduler
        list_of_downsamplers = [config.downsampling_config]
        list_of_thresholds = []

    return DownsamplingScheduler(
        modyn_config, pipeline_id, list_of_downsamplers, list_of_thresholds, config.maximum_keys_in_memory
    )
