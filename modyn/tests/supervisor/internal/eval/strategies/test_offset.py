from modyn.config.schema.pipeline import OffsetEvalStrategyConfig
from modyn.supervisor.internal.eval.strategies import OffsetEvalStrategy
from modyn.supervisor.internal.eval.strategies.abstract import EvalInterval


def test_get_eval_intervals():
    config = OffsetEvalStrategyConfig(offsets=["-inf", "inf", "0s", "100s", "-100s"])
    eval_strategy = OffsetEvalStrategy(config)
    intervals = list(eval_strategy.get_eval_intervals([(0, 0)]))
    assert intervals == [
        EvalInterval(start=0, end=0, training_interval_start=0, training_interval_end=0),
        EvalInterval(start=1, end=None, training_interval_start=0, training_interval_end=0),
        EvalInterval(start=0, end=1, training_interval_start=0, training_interval_end=0),
        EvalInterval(start=1, end=101, training_interval_start=0, training_interval_end=0),
        EvalInterval(start=0, end=0, training_interval_start=0, training_interval_end=0),
    ]

    intervals = list(eval_strategy.get_eval_intervals([(40, 70)]))
    assert intervals == [
        EvalInterval(start=0, end=40, training_interval_start=40, training_interval_end=70),
        EvalInterval(start=71, end=None, training_interval_start=40, training_interval_end=70),
        EvalInterval(start=40, end=71, training_interval_start=40, training_interval_end=70),
        EvalInterval(start=71, end=171, training_interval_start=40, training_interval_end=70),
        EvalInterval(start=0, end=40, training_interval_start=40, training_interval_end=70),
    ]

    intervals = list(eval_strategy.get_eval_intervals([(100, 100)]))
    assert intervals == [
        EvalInterval(start=0, end=100, training_interval_start=100, training_interval_end=100),
        EvalInterval(start=101, end=None, training_interval_start=100, training_interval_end=100),
        EvalInterval(start=100, end=101, training_interval_start=100, training_interval_end=100),
        EvalInterval(start=101, end=201, training_interval_start=100, training_interval_end=100),
        EvalInterval(start=0, end=100, training_interval_start=100, training_interval_end=100),
    ]

    intervals = list(eval_strategy.get_eval_intervals([(130, 200)]))
    assert intervals == [
        EvalInterval(start=0, end=130, training_interval_start=130, training_interval_end=200),
        EvalInterval(start=201, end=None, training_interval_start=130, training_interval_end=200),
        EvalInterval(start=130, end=201, training_interval_start=130, training_interval_end=200),
        EvalInterval(start=201, end=301, training_interval_start=130, training_interval_end=200),
        EvalInterval(start=30, end=130, training_interval_start=130, training_interval_end=200),
    ]
