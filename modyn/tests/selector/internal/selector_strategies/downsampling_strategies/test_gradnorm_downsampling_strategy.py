from modyn.selector.internal.selector_strategies.downsampling_strategies import GradNormDownsamplingStrategy


def test_init_gradnorm():
    # Test init works
    strat = GradNormDownsamplingStrategy(
        {
            "limit": -1,
            "reset_after_trigger": False,
            "presampling_ratio": 80,
            "downsampled_batch_size": 10,
        }
    )

    assert strat.downsampled_batch_size == 10
    assert isinstance(strat.get_downsampling_strategy(), str)
    assert strat.get_requires_remote_computation()


def test_command_gradnorm():
    # Test init works
    strat = GradNormDownsamplingStrategy(
        {
            "limit": -1,
            "reset_after_trigger": False,
            "presampling_ratio": 80,
            "downsampled_batch_size": 10,
        }
    )

    name = strat.get_downsampling_strategy()
    params = strat.get_downsampling_params()
    assert isinstance(name, str)
    assert name == "RemoteGradNormDownsampling"
    assert "downsampled_batch_size" in params
    assert params["downsampled_batch_size"] == 10
    assert strat.get_requires_remote_computation()
