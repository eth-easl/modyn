from modyn.config.schema.pipeline import OffsetEvalStrategyConfig
from modyn.supervisor.internal.eval.strategies import OffsetEvalStrategy
from modyn.supervisor.internal.eval.strategies.abstract import EvalInterval


def test_get_eval_intervals():
    config = OffsetEvalStrategyConfig(offsets=["-inf", "inf", "0s", "100s", "-100s"])
    eval_strategy = OffsetEvalStrategy(config)
    intervals = list(eval_strategy.get_eval_intervals([(0, 0)]))
    assert intervals == [
        EvalInterval(start=0, end=0, most_recent_model_interval_end_before=0),
        EvalInterval(start=1, end=None, most_recent_model_interval_end_before=0),
        EvalInterval(start=0, end=1, most_recent_model_interval_end_before=0),
        EvalInterval(start=1, end=101, most_recent_model_interval_end_before=0),
        EvalInterval(start=0, end=0, most_recent_model_interval_end_before=0),
    ]

    intervals = list(eval_strategy.get_eval_intervals([(40, 70)]))
    assert intervals == [
        EvalInterval(start=0, end=40, most_recent_model_interval_end_before=70),
        EvalInterval(start=71, end=None, most_recent_model_interval_end_before=70),
        EvalInterval(start=40, end=71, most_recent_model_interval_end_before=70),
        EvalInterval(start=71, end=171, most_recent_model_interval_end_before=70),
        EvalInterval(start=0, end=40, most_recent_model_interval_end_before=70),
    ]

    intervals = list(eval_strategy.get_eval_intervals([(100, 100)]))
    assert intervals == [
        EvalInterval(start=0, end=100, most_recent_model_interval_end_before=100),
        EvalInterval(start=101, end=None, most_recent_model_interval_end_before=100),
        EvalInterval(start=100, end=101, most_recent_model_interval_end_before=100),
        EvalInterval(start=101, end=201, most_recent_model_interval_end_before=100),
        EvalInterval(start=0, end=100, most_recent_model_interval_end_before=100),
    ]

    intervals = list(eval_strategy.get_eval_intervals([(130, 200)]))
    assert intervals == [
        EvalInterval(start=0, end=130, most_recent_model_interval_end_before=200),
        EvalInterval(start=201, end=None, most_recent_model_interval_end_before=200),
        EvalInterval(start=130, end=201, most_recent_model_interval_end_before=200),
        EvalInterval(start=201, end=301, most_recent_model_interval_end_before=200),
        EvalInterval(start=30, end=130, most_recent_model_interval_end_before=200),
    ]
