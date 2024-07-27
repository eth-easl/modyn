from modyn.config.schema.pipeline import OffsetEvalStrategyConfig
from modyn.supervisor.internal.eval.strategies import OffsetEvalStrategy
from modyn.supervisor.internal.eval.strategies.abstract import EvalInterval


def test_get_eval_intervals() -> None:
    config = OffsetEvalStrategyConfig(offsets=["-inf", "inf", "0s", "100s", "-100s"])
    eval_strategy = OffsetEvalStrategy(config)
    intervals = list(eval_strategy.get_eval_intervals([(0, 0)]))
    assert intervals == [
        EvalInterval(start=0, end=0),
        EvalInterval(start=1, end=None),
        EvalInterval(start=0, end=1),
        EvalInterval(start=1, end=101),
        EvalInterval(start=0, end=0),
    ]

    intervals = list(eval_strategy.get_eval_intervals([(40, 70)]))
    assert intervals == [
        EvalInterval(start=0, end=40),
        EvalInterval(start=71, end=None),
        EvalInterval(start=40, end=71),
        EvalInterval(start=71, end=171),
        EvalInterval(start=0, end=40),
    ]

    intervals = list(eval_strategy.get_eval_intervals([(100, 100)]))
    assert intervals == [
        EvalInterval(start=0, end=100),
        EvalInterval(start=101, end=None),
        EvalInterval(start=100, end=101),
        EvalInterval(start=101, end=201),
        EvalInterval(start=0, end=100),
    ]

    intervals = list(eval_strategy.get_eval_intervals([(130, 200)]))
    assert intervals == [
        EvalInterval(start=0, end=130),
        EvalInterval(start=201, end=None),
        EvalInterval(start=130, end=201),
        EvalInterval(start=201, end=301),
        EvalInterval(start=30, end=130),
    ]
