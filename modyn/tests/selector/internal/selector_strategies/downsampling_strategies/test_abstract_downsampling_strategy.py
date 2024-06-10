# pylint: disable=no-value-for-parameter,redefined-outer-name,abstract-class-instantiated
from unittest.mock import patch

from modyn.config.schema.pipeline.sampling.downsampling_config import BaseDownsamplingConfig
from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.utils import DownsamplingMode


@patch.multiple(AbstractDownsamplingStrategy, __abstractmethods__=set())
def test_constructor_invalid_config():
    modyn_config = {}
    pipeline_id = 0

    downsampling_config = BaseDownsamplingConfig(
        sample_then_batch=True,
        ratio=10,
        period=2,
    )

    ads = AbstractDownsamplingStrategy(downsampling_config, modyn_config, pipeline_id, 1000)

    assert ads.requires_remote_computation
    assert ads.downsampling_mode == DownsamplingMode.SAMPLE_THEN_BATCH
    assert ads.downsampling_ratio == 10
    assert ads.downsampling_period == 2
