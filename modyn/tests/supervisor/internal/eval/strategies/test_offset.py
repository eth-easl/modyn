from modyn.config.schema.pipeline import OffsetEvalStrategyConfig
from modyn.supervisor.internal.eval.strategies import OffsetEvalStrategy


def test_get_eval_intervals():
    config = OffsetEvalStrategyConfig(offsets=["-inf", "inf", "0s", "100s", "-100s"])
    eval_strategy = OffsetEvalStrategy(config)
    assert list(eval_strategy.get_eval_intervals(0, 0)) == [
        (0, 0),
        (1, None),
        (0, 1),
        (1, 101),
        (0, 0),
    ]

    assert list(eval_strategy.get_eval_intervals(40, 70)) == [
        (0, 40),
        (71, None),
        (40, 71),
        (71, 171),
        (0, 40),
    ]

    assert list(eval_strategy.get_eval_intervals(100, 100)) == [
        (0, 100),
        (101, None),
        (100, 101),
        (101, 201),
        (0, 100),
    ]

    assert list(eval_strategy.get_eval_intervals(130, 200)) == [
        (0, 130),
        (201, None),
        (130, 201),
        (201, 301),
        (30, 130),
    ]
