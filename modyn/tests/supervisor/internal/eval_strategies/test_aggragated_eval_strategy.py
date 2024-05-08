from modyn.supervisor.internal.eval_strategies import AggregatedEvalStrategy


def test_get_eval_interval():
    config = {"offsets": ["-inf", "inf", "0s", "100s", "-100s"]}
    eval_strategy = AggregatedEvalStrategy(config)
    assert list(eval_strategy.get_eval_interval(0, 0)) == [
        (0, 0),
        (1, -1),
        (0, 1),
        (1, 101),
        (0, 0),
    ]

    assert list(eval_strategy.get_eval_interval(40, 70)) == [
        (0, 40),
        (71, -1),
        (40, 71),
        (71, 171),
        (0, 40),
    ]

    assert list(eval_strategy.get_eval_interval(100, 100)) == [
        (0, 100),
        (101, -1),
        (100, 101),
        (101, 201),
        (0, 100),
    ]

    assert list(eval_strategy.get_eval_interval(130, 200)) == [
        (0, 130),
        (201, -1),
        (130, 201),
        (201, 301),
        (30, 130),
    ]
