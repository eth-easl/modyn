from __future__ import annotations

import pandas as pd
from modyn.config.schema.pipeline import EvalHandlerConfig
from modyn.supervisor.internal.eval.strategies.abstract import AbstractEvalStrategy, EvalInterval
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

        intervals = list(self.eval_strategy.get_eval_intervals([training_interval]))
        if not intervals:
            return []

        # first interval after the training interval
        most_recent_interval: None | EvalInterval = None
        for i in intervals:
            if training_interval[1] > (i.most_recent_model_interval_end_before or 0):
                continue  # constraint not met
            most_recent_interval = most_recent_interval or i
            if (i.most_recent_model_interval_end_before or i.end or i.start) <= (
                most_recent_interval.most_recent_model_interval_end_before
                or most_recent_interval.end
                or most_recent_interval.start
            ):
                most_recent_interval = i

        for i in intervals:
            for dataset_id in self.config.datasets:
                if self.config.models != "matrix" and (i != most_recent_interval):
                    continue
                eval_requests.append(
                    EvalRequest(
                        trigger_id=trigger_id,
                        training_id=training_id,
                        model_id=model_id,
                        most_recent_model=(i == most_recent_interval),
                        eval_handler=self.config.name,
                        dataset_id=dataset_id,
                        interval_start=i.start,
                        interval_end=i.end,
                    )
                )

        return eval_requests

    def get_eval_requests_after_pipeline(self, df_trainings: pd.DataFrame) -> list[EvalRequest]:
        """Args:
            df_trainings: The pipeline stage execution tracking information including training and model infos.
                Can be acquired by joining TriggerExecutionInfo and StoreModelInfo dataframes.

        Returns:
            list of evaluation requests with first and last timestamps being the point of trigger, not the interval.
        """
        assert {"trigger_id", "training_id", "id_model", "first_timestamp", "last_timestamp"} - (
            set(df_trainings.columns)
        ) == set()

        training_intervals: list[tuple[int, int]] = [
            (row["first_timestamp"], row["last_timestamp"]) for _, row in df_trainings.iterrows()
        ]
        eval_intervals = self.eval_strategy.get_eval_intervals(training_intervals)
        df_eval_intervals = pd.DataFrame(
            [
                (eval_interval.start, eval_interval.end, eval_interval.most_recent_model_interval_end_before)
                for eval_interval in eval_intervals
            ],
            columns=["start", "end", "most_recent_model_interval_end_before"],
        )

        # now build & potentially reduce the Trainings x EvalIntervals space
        df_cross = df_eval_intervals.merge(df_trainings, how="cross")

        # to check if a combination is the most recent, we first compute if model was trained before the usage starts
        # last_timestamp (df_trainings) defines the end of the training data;
        # most_recent_model_interval_end_before (df_eval_intervals): defines center of an eval intervals.
        df_cross["chronology_ok"] = df_cross["last_timestamp"] <= df_cross["most_recent_model_interval_end_before"]

        # find the maximum model for every EvalCandidate that doesn't violate that constraint
        max_model_id = (
            df_cross[df_cross["chronology_ok"]]
            .groupby("most_recent_model_interval_end_before")["id_model"]
            .aggregate(max_model_id="max")
        )

        # combine: a model in the cross product is most recent for a certain interval iff
        #  it has maximum model id for its most_recent_model_interval_end_before
        final = df_cross.merge(max_model_id, on="most_recent_model_interval_end_before")
        final["most_recent_model"] = final["id_model"] == final["max_model_id"]

        # convert to list
        eval_requests: list[EvalRequest] = []
        for _, eval_candidate in final.iterrows():
            if self.config.models != "matrix" and not eval_candidate["most_recent_model"]:
                continue
            for dataset_id in self.config.datasets:
                eval_requests.append(
                    EvalRequest(
                        trigger_id=eval_candidate["trigger_id"],
                        training_id=eval_candidate["training_id"],
                        model_id=eval_candidate["id_model"],
                        most_recent_model=eval_candidate["most_recent_model"],
                        eval_handler=self.config.name,
                        dataset_id=dataset_id,
                        interval_start=eval_candidate["start"],
                        interval_end=eval_candidate["end"],
                    )
                )

        return eval_requests
