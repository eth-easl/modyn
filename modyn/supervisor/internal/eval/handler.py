from __future__ import annotations

import pandas as pd
from modyn.config.schema.pipeline import EvalHandlerConfig
from modyn.supervisor.internal.eval.strategies.abstract import AbstractEvalStrategy
from modyn.utils import dynamic_module_import
from pydantic import BaseModel, Field

eval_strategy_module = dynamic_module_import("modyn.supervisor.internal.eval.strategies")


class EvalRequest(BaseModel):
    """A complete request for evaluation."""

    trigger_id: int
    training_id: int
    id_model: int
    currently_active_model: bool | None = Field(
        None,
        description=(
            "Whether the model is currently used for inference and can therefore be considered the active. "
            "This means that the evaluation interval starts after the model was trained. "
            "Only available for `after_pipeline` evaluations"
        ),
    )
    currently_trained_model: bool | None = Field(
        None,
        description=(
            "Whether the interval is in a range whether the model is currently trained on. "
            "For a fixed interval this is typically the model after the currently active model. "
            "Only available for `after_pipeline` evaluations"
        ),
    )
    eval_handler: str
    dataset_id: str
    interval_start: int = 0
    interval_end: int | None = None


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
        assert (
            self.config.models == "matrix"
        ), "Only matrix model evaluation is supported for after_training evaluations."
        eval_requests: list[EvalRequest] = []

        intervals = list(self.eval_strategy.get_eval_intervals([training_interval]))
        if not intervals:
            return []

        for i in intervals:
            for dataset_id in self.config.datasets:
                eval_requests.append(
                    EvalRequest(
                        trigger_id=trigger_id,
                        training_id=training_id,
                        id_model=model_id,
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
        df_trainings = df_trainings.copy()

        training_intervals: list[tuple[int, int]] = [
            (row["first_timestamp"], row["last_timestamp"]) for _, row in df_trainings.iterrows()
        ]
        eval_intervals = self.eval_strategy.get_eval_intervals(training_intervals)
        df_eval_intervals = pd.DataFrame(
            [
                (eval_interval.start, eval_interval.end, eval_interval.active_model_trained_before)
                for eval_interval in eval_intervals
            ],
            columns=["start", "end", "active_model_trained_before"],
        )

        # now build & potentially reduce the Trainings x EvalIntervals space
        df_cross = df_eval_intervals.merge(df_trainings, how="cross")

        # Check if a combination is the active. We first compute if model was trained before
        # the usage starts last_timestamp (df_trainings) defines the end of the training data;
        # active_model_trained_before (df_eval_intervals): defines center of an eval intervals.
        df_cross["active_candidate"] = df_cross["last_timestamp"] < df_cross["active_model_trained_before"]

        # find the maximum model for every EvalCandidate that doesn't violate that constraint
        max_model_id = (
            df_cross[df_cross["active_candidate"]]
            .groupby("active_model_trained_before")["id_model"]
            .aggregate(max_model_id="max")
        )

        # combine: a model in the cross product is most recent for a certain interval iff
        #  it has maximum model id for its active_model_trained_before
        df_active_models = df_cross.merge(max_model_id, on="active_model_trained_before", how="left")
        df_active_models["active_model"] = df_active_models["id_model"] == df_active_models["max_model_id"]

        # for a given interval, the currently trained model is the model with the smallest id
        # from all models that have a strictly bigger id than the most recent model. Hence it is the model after the
        # most recent model.
        # For that we first build a model -> successor model mapping:
        model_successor_relation = df_active_models[["id_model"]].drop_duplicates().sort_values(by="id_model")
        model_successor_relation["next_id_model"] = model_successor_relation["id_model"].shift(-1, fill_value=-1)

        # if there's no active model for the first interval(s), we still need to define the next model as the
        # trained model
        model_successor_relation = pd.concat(
            [
                model_successor_relation,
                pd.DataFrame([{"id_model": None, "next_id_model": df_active_models["id_model"].min()}]),
            ]
        )

        df_trained_models = df_active_models.merge(
            model_successor_relation, how="left", left_on="max_model_id", right_on="id_model", suffixes=("", "__")
        )
        df_trained_models["trained_model"] = df_trained_models["id_model"] == df_trained_models["next_id_model"]

        # convert to list
        eval_requests: list[EvalRequest] = []
        for _, eval_candidate in df_trained_models.iterrows():
            if (
                (self.config.models == "matrix")
                or (eval_candidate["active_model"] and self.config.models == "active")
                or (eval_candidate["trained_model"] and self.config.models == "train")
            ):
                for dataset_id in self.config.datasets:
                    eval_requests.append(
                        EvalRequest(
                            trigger_id=eval_candidate["trigger_id"],
                            training_id=eval_candidate["training_id"],
                            id_model=eval_candidate["id_model"],
                            currently_active_model=eval_candidate["active_model"],
                            currently_trained_model=eval_candidate["trained_model"],
                            eval_handler=self.config.name,
                            dataset_id=dataset_id,
                            interval_start=eval_candidate["start"],
                            interval_end=eval_candidate["end"],
                        )
                    )

        return eval_requests
