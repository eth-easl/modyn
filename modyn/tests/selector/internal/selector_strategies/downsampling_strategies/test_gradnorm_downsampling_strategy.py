from modyn.selector.internal.selector_strategies.downsampling_strategies import GradNormDownsamplingStrategy


def test_init_gradnorm():
    # Test init works
    strat = GradNormDownsamplingStrategy(
        {
            "ratio": 10,
            "sample_then_batch": True,
        },
        1000,
    )

    assert strat.downsampling_ratio == 10
    assert isinstance(strat.remote_downsampling_strategy_name, str)
    assert strat.requires_remote_computation


def test_command_gradnorm():
    # Test init works
    strat = GradNormDownsamplingStrategy(
        {
            "ratio": 10,
            "sample_then_batch": True,
        },
        1000,
    )

    name = strat.remote_downsampling_strategy_name
    params = strat.downsampling_params
    assert isinstance(name, str)
    assert name == "RemoteGradNormDownsampling"
    assert "downsampling_ratio" in params
    assert params["downsampling_ratio"] == 10
    assert strat.requires_remote_computation
    assert strat.maximum_keys_in_memory == 1000
