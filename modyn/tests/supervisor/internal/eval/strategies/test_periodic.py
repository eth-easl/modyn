from modyn.config.schema.pipeline import PeriodicEvalStrategyConfig
from modyn.supervisor.internal.eval.strategies.abstract import EvalInterval
from modyn.supervisor.internal.eval.strategies.periodic import PeriodicEvalStrategy


def test_periodic_eval_trigger() -> None:
    # test epoch based trigger
    config = PeriodicEvalStrategyConfig(every="5s", start_timestamp=0, end_timestamp=20, interval="[-2s, +2s]")
    strategy = PeriodicEvalStrategy(config)

    intervals = strategy.get_eval_intervals(
        training_intervals=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    )
    assert intervals == [
        EvalInterval(start=0, end=2, most_recent_model_interval_end_before=0),
        EvalInterval(start=3, end=7, most_recent_model_interval_end_before=5),
        EvalInterval(start=8, end=12, most_recent_model_interval_end_before=10),
        EvalInterval(start=13, end=17, most_recent_model_interval_end_before=15),
        EvalInterval(start=18, end=22, most_recent_model_interval_end_before=20),
    ]
