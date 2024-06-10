from modyn.config.schema.pipeline import SlicingEvalStrategyConfig
from modyn.supervisor.internal.eval.strategies import SlicingEvalStrategy
from modyn.supervisor.internal.eval.strategies.abstract import EvalInterval


def get_minimal_eval_strategies_config() -> SlicingEvalStrategyConfig:
    return SlicingEvalStrategyConfig(
        eval_every="100s",
        eval_start_from=0,
        eval_end_at=300,
    )


def test_get_eval_intervals() -> None:
    config = get_minimal_eval_strategies_config()
    eval_strategy = SlicingEvalStrategy(config)
    assert list(eval_strategy.get_eval_intervals([42, 99, 12])) == [
        EvalInterval(start=0, end=100, most_recent_model_interval_end_before=100),
        EvalInterval(start=100, end=200, most_recent_model_interval_end_before=200),
        EvalInterval(start=200, end=300, most_recent_model_interval_end_before=300),
    ]

    config.eval_start_from = 50
    config.eval_every = "60s"
    eval_strategy = SlicingEvalStrategy(config)
    assert list(eval_strategy.get_eval_intervals([42, 99, 12])) == [
        EvalInterval(start=50, end=110, most_recent_model_interval_end_before=110),
        EvalInterval(start=110, end=170, most_recent_model_interval_end_before=170),
        EvalInterval(start=170, end=230, most_recent_model_interval_end_before=230),
        EvalInterval(start=230, end=290, most_recent_model_interval_end_before=290),
        EvalInterval(start=290, end=300, most_recent_model_interval_end_before=300),
    ]
