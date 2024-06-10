from modyn.config.schema.pipeline.evaluation.strategy.between_two_triggers import BetweenTwoTriggersEvalStrategyConfig
from modyn.supervisor.internal.eval.strategies.abstract import EvalInterval
from modyn.supervisor.internal.eval.strategies.between_two_trainings import BetweenTwoTriggersEvalStrategy


def test_between_two_triggers():
    # test epoch based trigger
    config = BetweenTwoTriggersEvalStrategyConfig()
    strategy = BetweenTwoTriggersEvalStrategy(config)

    intervals = strategy.get_eval_intervals(training_intervals=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 6)])
    assert intervals == [
        EvalInterval(start=0, end=1, training_interval_start=0, training_interval_end=0),
        EvalInterval(start=1, end=2, training_interval_start=1, training_interval_end=1),
        EvalInterval(start=2, end=3, training_interval_start=2, training_interval_end=2),
        EvalInterval(start=3, end=4, training_interval_start=3, training_interval_end=3),
        EvalInterval(start=4, end=6, training_interval_start=5, training_interval_end=5),
    ]
