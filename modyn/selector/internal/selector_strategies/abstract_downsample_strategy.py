# pylint: disable=singleton-comparison
# flake8: noqa: E712
import logging

from modyn.selector.internal.selector_strategies.abstract_presample_strategy import AbstractPresampleStrategy
from modyn.utils import DownsamplingMode

logger = logging.getLogger(__name__)


class AbstractDownsampleStrategy(AbstractPresampleStrategy):
    """
    This abstract strategy is used to represent the common behaviour of downsampling strategies
    like loss-based, importance downsampling (distribution-based methods) and craig&adacore (greedy-based methods)

    These methods work on a uniformly-presampled version of the entire dataset (relying on AbstractPresampleStrategy),
    then the actual downsampling is done at the trainer server since all of these methods rely on the result of
    forward pass.

    Args:
        config (dict): The configuration for the selector.
    """

    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(config, modyn_config, pipeline_id, maximum_keys_in_memory)

        if "sample_then_batch" not in config:
            raise ValueError(
                "Please specify if you want to sample and then batch or vice versa. "
                "Use the sample_then_batch parameter"
            )
        if config["sample_then_batch"]:
            self.downsampling_mode = DownsamplingMode.SAMPLE_THEN_BATCH
        else:
            self.downsampling_mode = DownsamplingMode.BATCH_THEN_SAMPLE

        if self.downsampling_mode == DownsamplingMode.SAMPLE_THEN_BATCH and "downsampling_period" in config:
            raise ValueError("downsampling_period can be used only in sample-then-batch.")

        self.downsampling_period = config.get("downsampling_period", 1)

        if "downsampled_batch_ratio" not in config:
            raise ValueError("Please specify downsampled_batch_ratio to use downsampling methods")

        self.downsampled_batch_ratio = config["downsampled_batch_ratio"]

        if not (0 < self.downsampled_batch_ratio < 100) or not isinstance(self.downsampled_batch_ratio, int):
            raise ValueError("The downsampled batch ratio must be an integer in (0,100)")

        self._requires_remote_computation = True

    def get_downsampling_scale(self) -> float:
        if self.downsampling_mode == DownsamplingMode.SAMPLE_THEN_BATCH:
            return self.downsampled_batch_ratio / 100
        return 1

    def get_downsampler_config(self) -> dict:
        config = {
            "downsampled_batch_ratio": self.downsampled_batch_ratio,
            "maximum_keys_in_memory": self._maximum_keys_in_memory,
        }

        return config

    def get_trainer_config(self) -> dict:
        config = {
            "sample_then_batch": self.downsampling_mode == DownsamplingMode.SAMPLE_THEN_BATCH,
        }

        if self.downsampling_mode == DownsamplingMode.SAMPLE_THEN_BATCH:
            config["downsampling_period"] = self.downsampling_period

        return config
