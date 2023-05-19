# pylint: disable=singleton-comparison
# flake8: noqa: E712
import logging

from modyn.selector.internal.selector_strategies.abstract_presample_strategy import AbstractPresampleStrategy

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

        if "sample_before_batch" not in config:
            raise ValueError(
                "Please specify if you want to sample and then batch or vice versa. "
                "Use the sample_before_batch parameter"
            )
        self.sample_before_batch = config["sample_before_batch"]

        if self.sample_before_batch and "downsampling_period" in config:
            raise ValueError("downsampling_period can be used only in sample-then-batch.")
        self.downsampling_period = config.get("downsampling_period", 1)

        if self.sample_before_batch:
            # sample-then-batch, downsampled_batch_ratio is needed
            if "downsampled_batch_ratio" not in config:
                raise ValueError("Please specify downsampled_batch_ratio to use sample-then-batch")
            self.downsampled_batch_ratio = config["downsampled_batch_ratio"]
            if not (0 < self.downsampled_batch_ratio < 100) or not isinstance(self.downsampled_batch_ratio, int):
                raise ValueError("The downsampled batch ratio must be an integer in (0,100)")

        else:
            # batch-then-sample
            if "downsampled_batch_size" not in config:
                raise ValueError("Please specify downsampled_batch_size to use batch-then-sample")
            self.downsampled_batch_size = self._config["downsampled_batch_size"]
            if not isinstance(self.downsampled_batch_size, int):
                raise ValueError("The downsampled batch size must be an integer")

        self._requires_remote_computation = True
