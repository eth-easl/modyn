from __future__ import annotations

import logging
from functools import cached_property

from modyn.config.schema.pipeline.sampling.downsampling_config import UncertaintyDownsamplingConfig
from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.utils import DownsamplingMode

logger = logging.getLogger(__name__)


class TokenUncertaintyDownsamplingStrategy(AbstractDownsamplingStrategy):
    """
    Always-per-token uncertainty sampling for generative tasks.

    This strategy uses the same configuration as UncertaintyDownsamplingConfig,
    but forces a per-token selection mode (ignoring any weight_per_sample flag).
    """

    def __init__(
        self,
        downsampling_config: UncertaintyDownsamplingConfig,
        modyn_config: dict,
        pipeline_id: int,
        maximum_keys_in_memory: int,
    ):
        super().__init__(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)
        self.balance = downsampling_config.balance
        # Always use the token-level remote downsampler
        self.remote_downsampling_strategy_name = "RemoteTokenUncertaintyDownsampling"

    @cached_property
    def downsampling_params(self) -> dict:
        config = super().downsampling_params

        # Standard uncertainty params
        config["score_metric"] = self.downsampling_config.score_metric
        config["balance"] = self.balance
        # Force token-level sampling
        config["weight_per_sample"] = False

        if config["balance"] and self.downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE:
            raise ValueError("Balanced sampling (balance=True) can be used only in Sample then Batch mode.")

        return config
