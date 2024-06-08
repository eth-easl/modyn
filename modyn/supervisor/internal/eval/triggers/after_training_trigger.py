from __future__ import annotations

import pandas as pd
from modyn.supervisor.internal.eval.triggers.eval_trigger import EvalRequest, EvalTrigger
from typing_extensions import override


class AfterTrainingTrigger(EvalTrigger):

    def __init__(self) -> None:
        super().__init__(backlog=[])

    @override
    def get_eval_requests(self, df_trainings: pd.DataFrame, build_matrix: bool = True) -> list[EvalRequest]:
        assert {"trigger_id", "training_id", "id_model", "first_timestamp", "last_timestamp"} - (
            set(df_trainings.columns)
        ) == set()

        df_trainings = df_trainings.copy()

        assert build_matrix is False
        return [
            # we assume training duration 0 and use a new model
            # for all samples after the end of the training_batch (end_timestamp + 1)
            # until the end of next training batch (model_usage_end_timestamp, inclusive)
            EvalRequest(
                trigger_id=eval_candidate["trigger_id"],
                training_id=eval_candidate["training_id"],
                model_id=eval_candidate["id_model"],
                most_recent_model=True,
                interval_start=eval_candidate["first_timestamp"],
                interval_end=eval_candidate["last_timestamp"],
            )
            for _, eval_candidate in df_trainings.iterrows()
        ]
