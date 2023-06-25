# pylint: disable=singleton-comparison
# flake8: noqa: E712
import logging

from modyn.selector.internal.selector_strategies.general_presampling_strategy import GeneralPresamplingStrategy

logger = logging.getLogger(__name__)


class AbstractDownsampleStrategy(GeneralPresamplingStrategy):
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

        if "downsampled_batch_size" not in self._config:
            raise ValueError("To use downsampling strategies, you have to specify the downsampled_batch_size")
        self.downsampled_batch_size = self._config["downsampled_batch_size"]

        if not isinstance(self.downsampled_batch_size, int):
            raise ValueError("The downsampled batch size must be an integer")

        self._requires_remote_computation = True
