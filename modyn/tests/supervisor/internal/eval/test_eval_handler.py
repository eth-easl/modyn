from typing import Iterable

import pandas as pd
from modyn.config.schema.pipeline.evaluation.handler import EvalHandlerConfig
from modyn.config.schema.pipeline.evaluation.strategy.between_two_triggers import BetweenTwoTriggersEvalStrategyConfig
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
        EvalInterval(start=0, end=4, active_model_trained_before=5),
        EvalInterval(start=5, end=6, active_model_trained_before=6),
        EvalInterval(start=8, end=12, active_model_trained_before=7),
        EvalInterval(start=18, end=22, active_model_trained_before=20),
        EvalInterval(start=23, end=27, active_model_trained_before=25),
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
        (23, None, None, 0, 4),  # interval 1
        (23, None, None, 5, 6),  # interval 2
        (23, None, None, 8, 12),  # ...
        (23, None, None, 18, 22),
        (23, None, None, 23, 27),
    ]

    assert [
        (r.id_model, r.currently_active_model, r.currently_trained_model, r.interval_start, r.interval_end)
        for r in eval_requests
    ] == expected_eval_requests


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
        EvalInterval(start=8, end=12, active_model_trained_before=10),
        EvalInterval(start=18, end=22, active_model_trained_before=20),
        EvalInterval(start=23, end=27, active_model_trained_before=25),
        EvalInterval(start=24, end=28, active_model_trained_before=26),
        EvalInterval(start=25, end=29, active_model_trained_before=27),
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
    # (model_id, currently_active_model, currently_trained_model, start_interval, end_interval)
    expected_eval_requests = [
        # interval 1: for models/triggers 1-8
        (21, False, False, 8, 12),
        # checking the data above if looks like model 3 is the biggest model with
        # last_timestamp < active_model_trained_before;
        # however, we use next trainings start - 1 as training interval end.
        (22, True, False, 8, 12),
        (23, False, True, 8, 12),
        (26, False, False, 8, 12),
        (27, False, False, 8, 12),
        (28, False, False, 8, 12),
        # interval 2: for models/triggers 1-8
        (21, False, False, 18, 22),
        (22, False, False, 18, 22),
        (23, False, False, 18, 22),
        (26, True, False, 18, 22),
        (27, False, True, 18, 22),
        (28, False, False, 18, 22),
        # interval 3: for models/triggers 1-8
        (21, False, False, 23, 27),
        (22, False, False, 23, 27),
        (23, False, False, 23, 27),
        (26, True, False, 23, 27),
        (27, False, True, 23, 27),
        (28, False, False, 23, 27),
        # interval 4: for models/triggers 1-8
        (21, False, False, 24, 28),
        (22, False, False, 24, 28),
        (23, False, False, 24, 28),
        (26, False, False, 24, 28),
        (27, True, False, 24, 28),
        (28, False, True, 24, 28),
        # interval 5: for models/triggers 1-8
        (21, False, False, 25, 29),
        (22, False, False, 25, 29),
        (23, False, False, 25, 29),
        (26, False, False, 25, 29),
        (27, False, False, 25, 29),
        (28, True, False, 25, 29),
    ]

    assert [
        (r.id_model, r.currently_active_model, r.currently_trained_model, r.interval_start, r.interval_end)
        for r in eval_requests
    ] == expected_eval_requests

    # models=active
    eval_handler.config.models = "active"
    eval_requests = eval_handler.get_eval_requests_after_pipeline(trigger_dataframe)

    assert [
        (r.id_model, r.currently_active_model, r.currently_trained_model, r.interval_start, r.interval_end)
        for r in eval_requests
    ] == [exp for exp in expected_eval_requests if exp[1]]

    # models=train
    eval_handler.config.models = "train"
    eval_requests = eval_handler.get_eval_requests_after_pipeline(trigger_dataframe)

    assert [
        (r.id_model, r.currently_active_model, r.currently_trained_model, r.interval_start, r.interval_end)
        for r in eval_requests
    ] == [exp for exp in expected_eval_requests if exp[2]]


def test_between_two_trigger_after_pipeline() -> None:
    """Let's check for off-by-one errors"""

    trigger_dataframe = pd.DataFrame(
        {
            "trigger_id": [1, 4, 6, 9],
            "training_id": [11, 13, 18, 20],
            "id_model": [21, 26, 28, 29],
            "first_timestamp": [0, 5, 8, 14],
            "last_timestamp": [0, 7, 10, 16],
        }
    )

    eval_handler = EvalHandler(
        EvalHandlerConfig(
            strategy=BetweenTwoTriggersEvalStrategyConfig(),
            models="matrix",
            datasets=["dataset1"],
            execution_time="after_pipeline",
        )
    )
    eval_requests = eval_handler.get_eval_requests_after_pipeline(trigger_dataframe)

    # now assert the actual values in the full cross product (matrix mode)
    # (model_id, currently_active_model, currently_trained_model, start_interval, end_interval)
    expected_eval_requests = [
        # interval 1: for models/triggers 1-8
        (21, False, True, 0, 5 - 1),
        (26, False, False, 0, 5 - 1),
        (28, False, False, 0, 5 - 1),
        (29, False, False, 0, 5 - 1),
        # interval 2: for models/triggers 1-8
        (21, True, False, 5, 8 - 1),
        (26, False, True, 5, 8 - 1),
        (28, False, False, 5, 8 - 1),
        (29, False, False, 5, 8 - 1),
        # interval 3: for models/triggers 1-8
        (21, False, False, 8, 14 - 1),
        (26, True, False, 8, 14 - 1),
        (28, False, True, 8, 14 - 1),
        (29, False, False, 8, 14 - 1),
        # interval 4: for models/triggers 1-8
        (21, False, False, 14, 16),
        (26, False, False, 14, 16),
        (28, True, False, 14, 16),
        (29, False, True, 14, 16),
    ]

    assert [
        (r.id_model, r.currently_active_model, r.currently_trained_model, r.interval_start, r.interval_end)
        for r in eval_requests
    ] == expected_eval_requests

    # models=active
    eval_handler.config.models = "active"
    eval_requests = eval_handler.get_eval_requests_after_pipeline(trigger_dataframe)

    assert [
        (r.id_model, r.currently_active_model, r.currently_trained_model, r.interval_start, r.interval_end)
        for r in eval_requests
    ] == [exp for exp in expected_eval_requests if exp[1]]

    # models=train
    eval_handler.config.models = "train"
    eval_requests = eval_handler.get_eval_requests_after_pipeline(trigger_dataframe)

    assert [
        (r.id_model, r.currently_active_model, r.currently_trained_model, r.interval_start, r.interval_end)
        for r in eval_requests
    ] == [exp for exp in expected_eval_requests if exp[2]]
