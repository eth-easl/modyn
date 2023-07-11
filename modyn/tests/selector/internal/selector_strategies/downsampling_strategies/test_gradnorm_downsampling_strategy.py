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
    assert strat.maximum_keys_in_memory == 1000
    assert strat.requires_remote_computation

    name = strat.remote_downsampling_strategy_name
    assert isinstance(name, str)
    assert isinstance(name, str)
    assert name == "RemoteGradNormDownsampling"

    params = strat.downsampling_params
    assert "downsampling_ratio" in params
    assert params["downsampling_ratio"] == 10

