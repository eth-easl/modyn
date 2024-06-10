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
        EvalInterval(start=0, end=100, training_interval_start=50, training_interval_end=50),
        EvalInterval(start=100, end=200, training_interval_start=150, training_interval_end=150),
        EvalInterval(start=200, end=300, training_interval_start=250, training_interval_end=250),
    ]

    config.eval_start_from = 50
    config.eval_every = "60s"
    eval_strategy = SlicingEvalStrategy(config)
    assert list(eval_strategy.get_eval_intervals([42, 99, 12])) == [
        EvalInterval(start=50, end=110, training_interval_start=80, training_interval_end=80),
        EvalInterval(start=110, end=170, training_interval_start=140, training_interval_end=140),
        EvalInterval(start=170, end=230, training_interval_start=200, training_interval_end=200),
        EvalInterval(start=230, end=290, training_interval_start=260, training_interval_end=260),
        EvalInterval(start=290, end=300, training_interval_start=295, training_interval_end=295),
    ]
