from modyn.config.schema.pipeline.evaluation import IntervalEvalStrategyConfig
from modyn.supervisor.internal.eval.strategies.interval_eval_strategy import IntervalEvalStrategy


def test_get_eval_intervals() -> None:

    eval_strategy = IntervalEvalStrategy(IntervalEvalStrategyConfig(interval="[-inf,-0d]"))
    assert list(eval_strategy.get_eval_intervals(0, 0)) == [(0, 0)]
    assert list(eval_strategy.get_eval_intervals(40, 70)) == [(0, 40)]
    assert list(eval_strategy.get_eval_intervals(100, 100)) == [(0, 100)]
    assert list(eval_strategy.get_eval_intervals(130, 200)) == [(0, 130)]

    eval_strategy = IntervalEvalStrategy(IntervalEvalStrategyConfig(interval="(+0s, +inf]"))
    assert list(eval_strategy.get_eval_intervals(0, 0)) == [(1, None)]
    assert list(eval_strategy.get_eval_intervals(40, 70)) == [(71, None)]
    assert list(eval_strategy.get_eval_intervals(100, 100)) == [(101, None)]
    assert list(eval_strategy.get_eval_intervals(130, 200)) == [(201, None)]

    eval_strategy = IntervalEvalStrategy(IntervalEvalStrategyConfig(interval="[-0d, +0d]"))
    assert list(eval_strategy.get_eval_intervals(0, 0)) == [(0, 0)]
    assert list(eval_strategy.get_eval_intervals(40, 70)) == [(40, 70)]
    assert list(eval_strategy.get_eval_intervals(100, 100)) == [(100, 100)]
    assert list(eval_strategy.get_eval_intervals(130, 200)) == [(130, 200)]

    eval_strategy = IntervalEvalStrategy(IntervalEvalStrategyConfig(interval="(+0s, +100s]"))
    assert list(eval_strategy.get_eval_intervals(0, 0)) == [(1, 100)]
    assert list(eval_strategy.get_eval_intervals(40, 70)) == [(71, 170)]
    assert list(eval_strategy.get_eval_intervals(100, 100)) == [(101, 200)]
    assert list(eval_strategy.get_eval_intervals(130, 200)) == [(201, 300)]

    eval_strategy = IntervalEvalStrategy(IntervalEvalStrategyConfig(interval="[-100s, -0s]"))
    assert list(eval_strategy.get_eval_intervals(0, 0)) == [(0, 0)]
    assert list(eval_strategy.get_eval_intervals(40, 70)) == [(0, 40)]
    assert list(eval_strategy.get_eval_intervals(100, 100)) == [(0, 100)]
    assert list(eval_strategy.get_eval_intervals(130, 200)) == [(30, 130)]

    eval_strategy = IntervalEvalStrategy(IntervalEvalStrategyConfig(interval="(-15s, -10s)"))
    assert list(eval_strategy.get_eval_intervals(100, 200)) == [(86, 89)]
