from modyn.selector.internal.selector_strategies.downsampling_strategies import EmptyDownsamplingStrategy


def test_init_empty():
    # Test init works
    strat = EmptyDownsamplingStrategy(
        {},
        1000,
    )

    assert not strat.requires_remote_computation
    assert not strat.get_requires_remote_computation()

    assert strat.get_downsampling_strategy() == ""
    assert not strat.get_downsampling_params()  # strat.get_downsampling_params() == {} for pylint :(
