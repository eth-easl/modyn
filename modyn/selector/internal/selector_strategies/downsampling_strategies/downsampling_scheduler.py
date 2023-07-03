from typing import Optional, Tuple

from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.selector.internal.selector_strategies.downsampling_strategies.utils import instantiate_downsampler


class DownsamplingScheduler:
    def __init__(
        self, downsampling_configs: list[dict], downsampling_thresholds: list[int], maximum_keys_in_memory: int
    ):
        self.dowsampling_configs: list[dict] = downsampling_configs
        self.dowsampling_thresholds: list[int] = downsampling_thresholds
        self.downsampler_index = 0
        # transition from downsampler[i] to downsampler[i+1] happens before the thresholds[i]-th trigger

        self._validate_threshold()

        self.maximum_keys_in_memory = maximum_keys_in_memory

        self.current_downsampler, self.next_threshold = self._get_new_downsampler_and_threshold()

    def _validate_threshold(self) -> None:
        if not len(self.dowsampling_configs) == len(self.dowsampling_thresholds) + 1:
            raise ValueError("You must specify a threshold for each transition")
        if not sorted(self.dowsampling_thresholds) == self.dowsampling_thresholds:
            raise ValueError("Thresholds must be monotonically increasing")
        if not len(self.dowsampling_thresholds) == len(set(self.dowsampling_thresholds)):
            raise ValueError("Thresholds must be unique")

    def _get_new_downsampler_and_threshold(self) -> Tuple[AbstractDownsamplingStrategy, Optional[int]]:
        next_downsampler = instantiate_downsampler(
            self.dowsampling_configs[self.downsampler_index], self.maximum_keys_in_memory
        )
        # after instantiating the last downsampler, we simply set to none the next threshold
        next_threshold = (
            self.dowsampling_thresholds[self.downsampler_index]
            if self.downsampler_index < len(self.dowsampling_thresholds)
            else None
        )
        return next_downsampler, next_threshold

    def _update_downsampler_if_needed(self, next_trigger_id: int) -> None:
        if self.next_threshold is not None and next_trigger_id >= self.next_threshold:
            self.downsampler_index += 1
            self.current_downsampler, self.next_threshold = self._get_new_downsampler_and_threshold()

    def get_requires_remote_computation(self, next_trigger_id: int) -> bool:
        self._update_downsampler_if_needed(next_trigger_id)
        return self.current_downsampler.requires_remote_computation

    def get_downsampling_strategy(self, next_trigger_id: int) -> str:
        self._update_downsampler_if_needed(next_trigger_id)
        assert hasattr(
            self.current_downsampler, "remote_downsampling_strategy_name"
        ), "Your downsampler must specify the remote_downsampling_strategy_name"
        return self.current_downsampler.remote_downsampling_strategy_name

    def get_downsampling_params(self, next_trigger_id: int) -> dict:
        self._update_downsampler_if_needed(next_trigger_id)
        return self.current_downsampler.downsampling_params

    def get_training_status_bar_scale(self, next_trigger_id: int) -> int:
        self._update_downsampler_if_needed(next_trigger_id)
        return self.current_downsampler.status_bar_scale


def instantiate_scheduler(config: dict, maximum_keys_in_memory: int) -> DownsamplingScheduler:
    if "downsampling_config" not in config:
        # missing downsampler, use Empty
        list_of_downsamplers = [{"strategy": "EmptyDownsamplingStrategy"}]
        list_of_thresholds: list[int] = []
    elif "downsampling_list" not in config["downsampling_config"]:
        # just use one strategy, so fake scheduler
        list_of_downsamplers = [config["downsampling_config"]]
        list_of_thresholds = []
    else:
        # real scheduler
        list_of_downsamplers = config["downsampling_config"]["downsampling_list"]
        if "downsampling_thresholds" not in config["downsampling_config"]:
            raise ValueError(
                "You should specify the thresholds to switch from a downsampler to another. "
                "Use downsampling_thresholds"
            )
        list_of_thresholds = config["downsampling_config"]["downsampling_thresholds"]

        if isinstance(list_of_thresholds, int):
            list_of_thresholds = [list_of_thresholds]

    return DownsamplingScheduler(list_of_downsamplers, list_of_thresholds, maximum_keys_in_memory)
