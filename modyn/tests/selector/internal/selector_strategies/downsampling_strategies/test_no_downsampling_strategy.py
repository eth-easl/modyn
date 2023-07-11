from modyn.selector.internal.selector_strategies.downsampling_strategies import NoDownsamplingStrategy


def test_init_no():
    # Test init works
    strat = NoDownsamplingStrategy(
        {},
        1000,
    )

    assert not strat.requires_remote_computation
    assert strat.remote_downsampling_strategy_name == ""
