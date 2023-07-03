from modyn.selector.internal.selector_strategies.downsampling_strategies import EmptyDownsamplingStrategy


def test_init_empty():
    # Test init works
    strat = EmptyDownsamplingStrategy(
        {},
        1000,
    )

    assert not strat.requires_remote_computation

    assert strat.remote_downsampling_strategy_name == ""
