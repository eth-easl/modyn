from abc import ABC

from modyn.config import DownsamplingConfig
from modyn.selector.internal.storage_backend import AbstractStorageBackend
from modyn.utils import DownsamplingMode


class AbstractDownsamplingStrategy(ABC):
    """
    This abstract strategy is used to represent the common behaviour of downsampling strategies
    like loss-based, importance downsampling (distribution-based methods) and craig&adacore (greedy-based methods).

    These methods work on a uniformly-presampled version of the entire dataset (relying on AbstractPresampleStrategy),
    then the actual downsampling is done at the trainer server since all of these methods rely on the result of
    forward pass.

    If your downsampler requires remote computations, please make sure to specify the remote class using the parameter
    remote_downsampling_strategy_name. The value of the parameter should be the name of a class in the module
    modyn.trainer_server.internal.trainer.remote_downsamplers

    Args:
        downsampling_config (dict): The configuration for the selector.
    """

    def __init__(
        self, downsampling_config: DownsamplingConfig, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int
    ) -> None:
        self._modyn_config = modyn_config
        self._pipeline_id = pipeline_id
        if downsampling_config.sample_then_batch:
            self.downsampling_mode = DownsamplingMode.SAMPLE_THEN_BATCH
        else:
            self.downsampling_mode = DownsamplingMode.BATCH_THEN_SAMPLE

        self.downsampling_period = downsampling_config.period
        self.downsampling_ratio = downsampling_config.ratio

        self.requires_remote_computation = True
        self.maximum_keys_in_memory = maximum_keys_in_memory
        self.downsampling_config = downsampling_config
        self.downsampling_params = self._build_downsampling_params()
        self.status_bar_scale = self._compute_status_bar_scale()

    def _compute_status_bar_scale(self) -> int:
        """
        This function is used to create the downsampling status bar and handle the training one accordingly.

        For BTS, we return 100 since the training status bar sees all the samples
        For STB, we return the downsampling_ratio since the training status bar sees only a fraction of points
        (while the downsampling status bas sees all the points)
        """
        if self.downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE:
            return 100
        return self.downsampling_ratio

    def _build_downsampling_params(self) -> dict:
        config = {
            "downsampling_ratio": self.downsampling_ratio,
            "maximum_keys_in_memory": self.maximum_keys_in_memory,
            "sample_then_batch": self.downsampling_mode == DownsamplingMode.SAMPLE_THEN_BATCH,
        }

        if self.downsampling_mode == DownsamplingMode.SAMPLE_THEN_BATCH:
            config["downsampling_period"] = self.downsampling_period

        return config

    # pylint: disable=unused-argument
    def inform_next_trigger(self, next_trigger_id: int, selector_storage_backend: AbstractStorageBackend) -> None:
        """
        This function is used to inform the downsampler that the next trigger is reached.

        This is used for some downsamplers to implement some preparation logic before the actual downsampling
        on trainer server side happens, with the help of the argument `selector_storage_backend`.
        """

        # by default, no preparation is needed
        return
