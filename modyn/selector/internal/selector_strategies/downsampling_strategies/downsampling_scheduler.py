from typing import Tuple, Optional

from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.utils import dynamic_module_import


def instantiate_downsampler(config: dict, maximum_keys_in_memory: int) -> AbstractDownsamplingStrategy:
    downsampling_strategy_module = dynamic_module_import(
        "modyn.selector.internal.selector_strategies.downsampling_strategies"
    )
    if "downsampling_config" not in config or "strategy" not in config["downsampling_config"]:
        downsampling_strategy_name = "EmptyDownsamplingStrategy"
        downsampling_config = {}
    else:
        downsampling_strategy_name = config["downsampling_config"]["strategy"]
        downsampling_config = config["downsampling_config"]

    # for simplicity, you can just specify the short name (without DownsamplingStrategy)
    if not hasattr(downsampling_strategy_module, downsampling_strategy_name):
        long_name = f"{downsampling_strategy_name}DownsamplingStrategy"
        if not hasattr(downsampling_strategy_module, long_name):
            raise ValueError("Requested downsampling strategy does not exist")
        downsampling_strategy_name = long_name

    downsampling_class = getattr(downsampling_strategy_module, downsampling_strategy_name)
    return downsampling_class(downsampling_config, maximum_keys_in_memory)


class DownsamplingScheduler:
    def __init__(
        self, downsampling_configs: list[dict], downsampling_thresholds: list[int], maximum_keys_in_memory: int
    ):
        self.dowsampling_configs: list[dict] = downsampling_configs
        self.dowsampling_thresholds: list[int] = downsampling_thresholds
        self.downsampler_index = 0
        # transition from downsampler[i] to downsampler[i+1] happens before the thresholds[i]-th trigger

        assert (
            len(self.dowsampling_configs) == len(self.dowsampling_thresholds) + 1
        ), "You must specify a threshold for each transition"

        assert (
            sorted(self.dowsampling_thresholds) == self.dowsampling_thresholds
        ), "Thresholds must be monotonically increasing"

        self.maximum_keys_in_memory = maximum_keys_in_memory

        self.current_downsampler, self.next_threshold = self._get_new_downsampler_and_threshold()

    def _get_new_downsampler_and_threshold(self) -> Tuple[AbstractDownsamplingStrategy, Optional[int]]:
        next_downsampler = instantiate_downsampler(
            self.dowsampling_configs[self.downsampler_index], self.maximum_keys_in_memory
        )
        # after instantiating the last downsampler, we simply set to none the next threshold
        next_threshold = (
            self.dowsampling_thresholds[self.downsampler_index]
            if len(self.dowsampling_thresholds) < self.downsampler_index
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
