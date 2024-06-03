from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class EvalCandidate:
    """Indicates a point in time where an EvalRequest should be issued.

    EvalRequest is not created directly as model_id and trigger_id is not known at this point.
    """

    sample_timestamp: int
    sample_index: int | None = None
    sample_id: int | None = None


@dataclass
class EvalRequest:
    """A complete request for evaluation."""

    trigger_id: int
    training_id: int
    model_id: int
    most_recent_model: bool
    interval_start: int | None
    interval_end: int | None


class EvalTrigger:

    def __init__(self) -> None:
        self.evaluation_backlog: list[EvalCandidate] = []

    def inform(self, new_data: list[tuple[int, int, int]]) -> None:
        """Inform the trigger about a batch of new data.

        Side Effects:
            The trigger appends EvalRequests internally for  all points in time where evaluation is required.
        """

    def inform_about_training(
        self, trigger_id: int, training_id: int, model_id: int, start_timestamp: int, end_timestamp: int
    ) -> None:
        """Inform the trigger about a training event. Not used in all triggers."""

    def get_eval_requests(self, df_trainings: pd.DataFrame, build_matrix: bool = True) -> list[EvalRequest]:
        """
        Args:
            df_store_model: The pipeline stage execution tracking information including training and model infos.
                Can be acquired by joining TriggerExecutionInfo and StoreModelInfo dataframes.
            build_matrix: weather to evaluate every model on every evaluation EvalCandidate time.

        Returns:
            list of evaluation requests with first and last timestamps being the point of trigger, not the interval.
        """
        assert {"trigger_id", "training_id", "id_model", "first_timestamp", "last_timestamp"} - (
            set(df_trainings.columns)
        ) == set()

        # join candidates and actual traings
        df_candidates = pd.DataFrame(
            [c.sample_timestamp for c in self.evaluation_backlog], columns=["sample_timestamp"]
        )
        df_cross = df_candidates.merge(df_trainings, how="cross")

        # to check if a combination is the most recent we first compute if model was trained before the usage starts
        df_cross["chronology_ok"] = df_cross["last_timestamp"] <= df_cross["sample_timestamp"]

        # find the maximum model for every EvalCandidate that doesn't violate that constraint
        max_model_id = (
            df_cross[df_cross["chronology_ok"]].groupby("sample_timestamp")["id_model"].aggregate(max_model_id="max")
        )

        # combine: an element in the cross product is most recent iff it has maximum model id for its sample_timestamp
        final = df_cross.merge(max_model_id, on="sample_timestamp")
        final["most_recent_model"] = final["id_model"] == final["max_model_id"]

        # convert to list
        eval_requests: list[EvalRequest] = []
        for _, eval_candidate in final.iterrows():
            if not build_matrix and not eval_candidate["most_recent_model"]:
                continue
            eval_requests.append(
                EvalRequest(
                    trigger_id=eval_candidate["trigger_id"],
                    training_id=eval_candidate["training_id"],
                    model_id=eval_candidate["id_model"],
                    most_recent_model=eval_candidate["most_recent_model"],
                    interval_start=eval_candidate["sample_timestamp"],
                    interval_end=eval_candidate["sample_timestamp"],
                )
            )

        return eval_requests
