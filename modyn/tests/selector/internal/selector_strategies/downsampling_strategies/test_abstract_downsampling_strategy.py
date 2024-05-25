# pylint: disable=no-value-for-parameter,redefined-outer-name,abstract-class-instantiated
from unittest.mock import patch

import pytest
from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy


@patch.multiple(AbstractDownsamplingStrategy, __abstractmethods__=set())
def test_constructor_invalid_config():
    # missing downsampling_ratio
    downsampling_config = {
        "sample_then_batch": True,
    }
    modyn_config = {}
    pipeline_id = 0
    with pytest.raises(ValueError):
        AbstractDownsamplingStrategy(downsampling_config, modyn_config, pipeline_id, 1000)

    # float downsampling_ratio
    downsampling_config = {"sample_then_batch": True, "ratio": 0.18}
    with pytest.raises(ValueError):
        AbstractDownsamplingStrategy(downsampling_config, modyn_config, pipeline_id, 1000)

    downsampling_config = {"sample_then_batch": True, "ratio": 10}

    ads = AbstractDownsamplingStrategy(downsampling_config, modyn_config, pipeline_id, 1000)

    assert ads.requires_remote_computation
    assert ads.downsampling_ratio == 10
