from modyn.selector.internal.selector_strategies.downsampling_strategies import GradNormDownsamplingStrategy


def test_init_gradnorm():
    # Test init works
    strat = GradNormDownsamplingStrategy(
        {
            "limit": -1,
            "reset_after_trigger": False,
            "presampling_ratio": 80,
            "downsampling_ratio": 10,
            "sample_then_batch": True,
        },
        1000,
    )

    assert strat.downsampling_ratio == 10
    assert isinstance(strat.get_downsampling_strategy(), str)
    assert strat.get_requires_remote_computation()


def test_command_gradnorm():
    # Test init works
    strat = GradNormDownsamplingStrategy(
        {
            "limit": -1,
            "reset_after_trigger": False,
            "presampling_ratio": 80,
            "downsampling_ratio": 10,
            "sample_then_batch": True,
        },
        1000,
    )

    name = strat.get_downsampling_strategy()
    params = strat.get_downsampling_params()
    assert isinstance(name, str)
    assert name == "RemoteGradNormDownsampling"
    assert "downsampling_ratio" in params
    assert params["downsampling_ratio"] == 10
    assert strat.get_requires_remote_computation()
    assert strat.maximum_keys_in_memory == 1000
