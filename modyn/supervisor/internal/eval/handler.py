from __future__ import annotations

import pandas as pd
from modyn.config.schema.pipeline import EvalHandlerConfig
from modyn.supervisor.internal.eval.strategies.abstract import AbstractEvalStrategy
from modyn.utils import dynamic_module_import
from pydantic import BaseModel

eval_strategy_module = dynamic_module_import("modyn.supervisor.internal.eval.strategies")


class EvalRequest(BaseModel):
    """A complete request for evaluation."""

    trigger_id: int
    training_id: int
    model_id: int
    most_recent_model: bool
    eval_handler: str
    dataset_id: str
    interval_start: int | None
    interval_end: int | None


class EvalHandler:
    """Handles one series of evaluations configured through an `EvalHandlerConfig`."""

    def __init__(self, config: EvalHandlerConfig):
        self.config = config
        strat_conf = config.strategy
        self.eval_strategy: AbstractEvalStrategy = getattr(eval_strategy_module, strat_conf.type)(strat_conf)

    def get_eval_requests_after_training(
        self, trigger_id: int, training_id: int, model_id: int, training_interval: tuple[int, int]
    ) -> list[EvalRequest]:
        """Returns a list of evaluation requests for the current handler."""
        eval_requests: list[EvalRequest] = []

        intervals = self.eval_strategy.get_eval_intervals([training_interval])

        most_recent_found: bool = False
        for interval in intervals:
            # first interval that starts after the training interval
            is_most_recent = not most_recent_found and ((interval.start or 0) >= training_interval[1])
            most_recent_found = most_recent_found or is_most_recent
            for dataset_id in self.config.datasets:
                eval_requests.append(
                    EvalRequest(
                        trigger_id=trigger_id,
                        training_id=training_id,
                        model_id=model_id,
                        most_recent_model=is_most_recent,
                        eval_handler=self.config.name,
                        dataset_id=dataset_id,
                        interval_start=interval.start,
                        interval_end=interval.end,
                    )
                )

        return eval_requests

    def get_eval_requests_after_pipeline(self, df_trainings: pd.DataFrame) -> list[EvalRequest]:
        """Args:
            df_store_model: The pipeline stage execution tracking information including training and model infos.
                Can be acquired by joining TriggerExecutionInfo and StoreModelInfo dataframes.

        Returns:
            list of evaluation requests with first and last timestamps being the point of trigger, not the interval.
        """
        assert {"trigger_id", "training_id", "id_model", "first_timestamp", "last_timestamp"} - (
            set(df_trainings.columns)
        ) == set()

        return []  # followup-PR
