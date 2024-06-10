from modyn.config.schema.pipeline.evaluation.strategy.between_two_triggers import BetweenTwoTriggersEvalStrategyConfig
from modyn.supervisor.internal.eval.strategies.abstract import EvalInterval
from modyn.supervisor.internal.eval.strategies.between_two_triggers import BetweenTwoTriggersEvalStrategy


def test_between_two_triggers():
    # test epoch based trigger
    config = BetweenTwoTriggersEvalStrategyConfig()
    strategy = BetweenTwoTriggersEvalStrategy(config)

    intervals = strategy.get_eval_intervals(training_intervals=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 6)])
    assert intervals == [
        EvalInterval(start=0, end=1, most_recent_model_interval_end_before=1),
        EvalInterval(start=1, end=2, most_recent_model_interval_end_before=2),
        EvalInterval(start=2, end=3, most_recent_model_interval_end_before=3),
        EvalInterval(start=3, end=4, most_recent_model_interval_end_before=4),
        EvalInterval(start=4, end=6, most_recent_model_interval_end_before=6),
    ]
