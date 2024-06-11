from typing import Iterable

import pandas as pd
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig
from modyn.config.schema.pipeline.evaluation.strategy.static import StaticEvalStrategyConfig
from modyn.supervisor.internal.eval.handler import EvalHandler
from modyn.supervisor.internal.eval.strategies.abstract import AbstractEvalStrategy, EvalInterval


class DummyEvalStrategy(AbstractEvalStrategy):
    def __init__(self, config: StaticEvalStrategyConfig, intervals: Iterable[tuple[int, int]]):
        super().__init__(config)
        self.intervals = intervals

    def get_eval_intervals(self, training_intervals: Iterable[tuple[int, int]]) -> Iterable[EvalInterval]:
        return self.intervals


def test_get_eval_requests_after_training() -> None:
    trigger_id = 3
    training_id = 13
    model_id = 23
    training_interval = (5, 7)

    intervals = [
        EvalInterval(start=0, end=4, most_recent_model_interval_end_before=5),
        EvalInterval(start=5, end=6, most_recent_model_interval_end_before=6),
        EvalInterval(start=8, end=12, most_recent_model_interval_end_before=7),
        EvalInterval(start=18, end=22, most_recent_model_interval_end_before=20),
        EvalInterval(start=23, end=27, most_recent_model_interval_end_before=25),
    ]

    eval_handler = EvalHandler(
        EvalHandlerConfig(
            strategy=StaticEvalStrategyConfig(intervals=[]),
            models="matrix",
            datasets=["dataset1"],
            execution_time="after_training",
        )
    )
    eval_handler.eval_strategy = DummyEvalStrategy(eval_handler.config, intervals)
    eval_requests = eval_handler.get_eval_requests_after_training(trigger_id, training_id, model_id, training_interval)

    # only consider current model 23
    expected_eval_requests = [
        (23, False, 0, 4),  # interval 1
        (23, False, 5, 6),  # interval 2
        (23, True, 8, 12),  # ...
        (23, False, 18, 22),
        (23, False, 23, 27),
    ]

    assert [
        (r.model_id, r.most_recent_model, r.interval_start, r.interval_end) for r in eval_requests
    ] == expected_eval_requests

    # models=most_recent
    eval_handler.config.models = "most_recent"
    eval_requests = eval_handler.get_eval_requests_after_training(trigger_id, training_id, model_id, training_interval)

    assert [(r.model_id, r.most_recent_model, r.interval_start, r.interval_end) for r in eval_requests] == [
        exp for exp in expected_eval_requests if exp[1]
    ]


def test_get_eval_requests_after_pipeline() -> None:
    trigger_dataframe = pd.DataFrame(
        {
            "trigger_id": [1, 2, 3, 6, 7, 8],
            "training_id": [11, 12, 13, 16, 17, 18],
            "id_model": [21, 22, 23, 26, 27, 28],
            "first_timestamp": [0, 2, 3, 14, 20, 26],
            "last_timestamp": [0, 2, 3, 14, 20, 26],
        }
    )

    intervals = [
        EvalInterval(start=8, end=12, most_recent_model_interval_end_before=10),
        EvalInterval(start=18, end=22, most_recent_model_interval_end_before=20),
        EvalInterval(start=23, end=27, most_recent_model_interval_end_before=25),
        EvalInterval(start=24, end=28, most_recent_model_interval_end_before=26),
        EvalInterval(start=25, end=29, most_recent_model_interval_end_before=27),
    ]

    eval_handler = EvalHandler(
        EvalHandlerConfig(
            strategy=StaticEvalStrategyConfig(intervals=[]),
            models="matrix",
            datasets=["dataset1"],
            execution_time="after_pipeline",
        )
    )
    eval_handler.eval_strategy = DummyEvalStrategy(eval_handler.config, intervals)
    eval_requests = eval_handler.get_eval_requests_after_pipeline(trigger_dataframe)

    # now assert the actual values in the full cross product (matrix mode)
    # (model_id, most_recent_model, start_interval, end_interval)
    expected_eval_requests = [
        # interval 1: for models/triggers 1-8
        (21, False, 8, 12),
        (22, False, 8, 12),
        (23, True, 8, 12),
        (26, False, 8, 12),
        (27, False, 8, 12),
        (28, False, 8, 12),
        # interval 2: for models/triggers 1-8
        (21, False, 18, 22),
        (22, False, 18, 22),
        (23, False, 18, 22),
        (26, False, 18, 22),
        (27, True, 18, 22),
        (28, False, 18, 22),
        # interval 3: for models/triggers 1-8
        (21, False, 23, 27),
        (22, False, 23, 27),
        (23, False, 23, 27),
        (26, False, 23, 27),
        (27, True, 23, 27),
        (28, False, 23, 27),
        # interval 4: for models/triggers 1-8
        (21, False, 24, 28),
        (22, False, 24, 28),
        (23, False, 24, 28),
        (26, False, 24, 28),
        (27, False, 24, 28),
        (28, True, 24, 28),
        # interval 5: for models/triggers 1-8
        (21, False, 25, 29),
        (22, False, 25, 29),
        (23, False, 25, 29),
        (26, False, 25, 29),
        (27, False, 25, 29),
        (28, True, 25, 29),
    ]

    assert [
        (r.model_id, r.most_recent_model, r.interval_start, r.interval_end) for r in eval_requests
    ] == expected_eval_requests

    # models=most_recent
    eval_handler.config.models = "most_recent"
    eval_requests = eval_handler.get_eval_requests_after_pipeline(trigger_dataframe)

    assert [(r.model_id, r.most_recent_model, r.interval_start, r.interval_end) for r in eval_requests] == [
        exp for exp in expected_eval_requests if exp[1]
    ]
