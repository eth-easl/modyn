from modyn.config.schema.pipeline.evaluation.strategy.between_two_triggers import BetweenTwoTriggersEvalStrategyConfig
from modyn.supervisor.internal.eval.strategies.abstract import EvalInterval
from modyn.supervisor.internal.eval.strategies.between_two_triggers import BetweenTwoTriggersEvalStrategy


def test_between_two_triggers() -> None:
    # test epoch based trigger
    config = BetweenTwoTriggersEvalStrategyConfig()
    strategy = BetweenTwoTriggersEvalStrategy(config)

    intervals = strategy.get_eval_intervals(
        training_intervals=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)], dataset_end_time=12
    )
    assert intervals == [
        EvalInterval(start=0, end=1, active_model_trained_before=0),
        EvalInterval(start=2, end=3, active_model_trained_before=2),
        EvalInterval(start=4, end=5, active_model_trained_before=4),
        EvalInterval(start=6, end=7, active_model_trained_before=6),
        EvalInterval(start=8, end=9, active_model_trained_before=8),
        EvalInterval(start=10, end=12, active_model_trained_before=10),
    ]
