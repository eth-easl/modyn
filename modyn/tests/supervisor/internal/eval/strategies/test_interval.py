from modyn.config.schema.pipeline.evaluation.strategy._interval import _IntervalEvalStrategyConfig
from modyn.supervisor.internal.eval.strategies._interval import _IntervalEvalStrategyMixin


def test_generate_interval() -> None:
    eval_strategy = _IntervalEvalStrategyMixin(_IntervalEvalStrategyConfig(interval="[-inf,-0d]"))
    assert eval_strategy._generate_interval(0, 0) == (0, 0)
    assert eval_strategy._generate_interval(40, 70) == (0, 40)
    assert eval_strategy._generate_interval(100, 100) == (0, 100)
    assert eval_strategy._generate_interval(130, 200) == (0, 130)

    eval_strategy = _IntervalEvalStrategyMixin(_IntervalEvalStrategyConfig(interval="(+0s, +inf]"))
    assert eval_strategy._generate_interval(0, 0) == (1, None)
    assert eval_strategy._generate_interval(40, 70) == (71, None)
    assert eval_strategy._generate_interval(100, 100) == (101, None)
    assert eval_strategy._generate_interval(130, 200) == (201, None)

    eval_strategy = _IntervalEvalStrategyMixin(_IntervalEvalStrategyConfig(interval="[-0d, +0d]"))
    assert eval_strategy._generate_interval(0, 0) == (0, 0)
    assert eval_strategy._generate_interval(40, 70) == (40, 70)
    assert eval_strategy._generate_interval(100, 100) == (100, 100)
    assert eval_strategy._generate_interval(130, 200) == (130, 200)

    eval_strategy = _IntervalEvalStrategyMixin(_IntervalEvalStrategyConfig(interval="(+0s, +100s]"))
    assert eval_strategy._generate_interval(0, 0) == (1, 100)
    assert eval_strategy._generate_interval(40, 70) == (71, 170)
    assert eval_strategy._generate_interval(100, 100) == (101, 200)
    assert eval_strategy._generate_interval(130, 200) == (201, 300)

    eval_strategy = _IntervalEvalStrategyMixin(_IntervalEvalStrategyConfig(interval="[-100s, -0s]"))
    assert eval_strategy._generate_interval(0, 0) == (0, 0)
    assert eval_strategy._generate_interval(40, 70) == (0, 40)
    assert eval_strategy._generate_interval(100, 100) == (0, 100)
    assert eval_strategy._generate_interval(130, 200) == (30, 130)

    eval_strategy = _IntervalEvalStrategyMixin(_IntervalEvalStrategyConfig(interval="(-15s, -10s)"))
    assert eval_strategy._generate_interval(100, 200) == (86, 89)
