from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from modyn.supervisor.internal.eval.triggers.eval_trigger import EvalRequest, EvalTrigger
from typing_extensions import override


@dataclass
class _TrainingLog:
    trigger_id: int
    training_id: int
    model_id: int

    start_timestamp: int
    end_timestamp: int
    model_usage_end_timestamp: int | None


class AfterTrainingTrigger(EvalTrigger):

    def __init__(self) -> None:
        super().__init__()
        self.trainings: list[_TrainingLog] = []

    def inform_about_training(
        self, trigger_id: int, training_id: int, model_id: int, start_timestamp: int, end_timestamp: int
    ) -> None:
        if len(self.trainings) > 0:
            assert self.trainings[-1].model_usage_end_timestamp is None
            self.trainings[-1].model_usage_end_timestamp = end_timestamp
        self.trainings.append(
            _TrainingLog(
                trigger_id=trigger_id,
                training_id=training_id,
                model_id=model_id,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                model_usage_end_timestamp=None,
            )
        )

    @override
    def get_eval_requests(self, df_trainings: pd.DataFrame, build_matrix: bool = False) -> list[EvalRequest]:
        assert self.trainings[-1].model_usage_end_timestamp is None
        return [
            # we assume training duration 0 and use a new model
            # for all samples after the end of the training_batch (end_timestamp + 1)
            # until the end of next training batch (model_usage_end_timestamp, inclusive)
            EvalRequest(
                trigger_id=training.trigger_id,
                training_id=training.training_id,
                model_id=training.model_id,
                most_recent_model=True,
                interval_start=training.end_timestamp + 1,
                interval_end=training.model_usage_end_timestamp,
            )
            for training in self.trainings
            if training.model_usage_end_timestamp is not None
        ]
