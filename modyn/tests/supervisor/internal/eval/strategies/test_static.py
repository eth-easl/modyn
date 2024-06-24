from modyn.config.schema.pipeline import StaticEvalStrategyConfig
from modyn.supervisor.internal.eval.strategies.abstract import EvalInterval
from modyn.supervisor.internal.eval.strategies.static import StaticEvalStrategy


def test_static_eval_trigger() -> None:
    # test epoch based trigger
    config = StaticEvalStrategyConfig(intervals=[(1, 2), (2, 2), (0, 8)])
    strategy = StaticEvalStrategy(config)

    intervals = strategy.get_eval_intervals(
        training_intervals=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    )
    assert intervals == [
        EvalInterval(start=1, end=2, active_model_trained_before=1),
        EvalInterval(start=2, end=2, active_model_trained_before=2),
        EvalInterval(start=0, end=8, active_model_trained_before=0),
    ]
