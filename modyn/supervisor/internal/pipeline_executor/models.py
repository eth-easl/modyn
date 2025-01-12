from __future__ import annotations

import dataclasses
import datetime
import itertools
import logging
import multiprocessing as mp
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Union, cast

import pandas as pd
from pydantic import BaseModel, Field, model_serializer, model_validator
from typing_extensions import override

from modyn.config.schema.pipeline import ModynPipelineConfig
from modyn.config.schema.system.config import ModynConfig
from modyn.supervisor.internal.eval.handler import EvalRequest
from modyn.supervisor.internal.grpc.enums import PipelineStage
from modyn.supervisor.internal.triggers.utils.models import TriggerPolicyEvaluationLog
from modyn.supervisor.internal.utils.evaluation_status_reporter import (
    EvaluationStatusReporter,
)
from modyn.supervisor.internal.utils.git_utils import get_head_sha

logger = logging.getLogger(__name__)


@dataclass
class PipelineExecutionParams:
    """Wrapped initialization options for the pipeline executor, many of them
    are cli arguments."""

    start_timestamp: int
    pipeline_id: int
    modyn_config: ModynConfig
    pipeline_config: ModynPipelineConfig
    eval_directory: Path
    exception_queue: mp.Queue
    pipeline_status_queue: mp.Queue
    training_status_queue: mp.Queue
    eval_status_queue: mp.Queue
    start_replay_at: int | None = None
    stop_replay_at: int | None = None
    maximum_triggers: int | None = None

    @property
    def dataset_id(self) -> str:
        return self.pipeline_config.data.dataset_id

    @property
    def log_directory(self) -> Path:
        return self.eval_directory

    @property
    def experiment_mode(self) -> bool:
        return self.start_replay_at is not None

    @cached_property
    def selector_batch_size(self) -> int:
        for dataset in self.modyn_config.storage.datasets:
            if dataset.name == self.dataset_id:
                return dataset.selector_batch_size

        logger.info(
            f"Dataset with id {self.dataset_id} not found in storage configuration. "
            "Using default 128 for selector_batch_size."
        )
        return 128

    @model_validator(mode="after")
    def validate_replay(self) -> PipelineExecutionParams:
        if self.start_replay_at is None and self.stop_replay_at is not None:
            raise ValueError("`stop_replay_at` can only be used in conjunction with `start_replay_at`.")
        return self

    @property
    def pipeline_logdir(self) -> Path:
        return self.log_directory / f"pipeline_{self.pipeline_id}"


@dataclass
class PipelineBatchState:
    """Pipeline artifacts shared across stages during the processing of one
    batch."""

    data: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)
    remaining_data: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)

    triggering_indexes: list[int] = dataclasses.field(default_factory=list)
    previous_trigger_idx: int = 0

    trigger_index: int = -1
    """The index in the dataset where the trigger was initiated."""

    trigger_id: int = -1
    """The identifier of the trigger received from the selector."""

    evaluations: dict[int, EvaluationStatusReporter] = dataclasses.field(default_factory=dict)


@dataclass
class ExecutionState(PipelineExecutionParams):
    """Represent the state of the pipeline executor including artifacts of
    pipeline stages."""

    stage: PipelineStage = PipelineStage.INIT
    """The current stage of the pipeline executor."""

    stage_id_seq_counters: dict[str, int] = dataclasses.field(default_factory=dict)
    """Tracks for every stage that can be logged in StageLog how many logs have
    been created using this id.

    This information can be used to uniquely identify logs over multiple
    pipeline runs given the pipelines use a deterministic configuration.
    """

    # for logging
    seen_pipeline_stages: set[PipelineStage] = dataclasses.field(default_factory=set)
    current_batch_index: int = 0
    current_sample_index: int = 0
    current_sample_time: int = 0  # unix timestamp
    """The unix timestamp of the last sample seen and processed by the pipeline
    executor."""

    # this is to store the first and last timestamp of the remaining data after handling all triggers
    remaining_data: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)
    remaining_data_range: tuple[int, int] | None = None

    triggers: list[int] = dataclasses.field(default_factory=list)
    current_training_id: int | None = None
    trained_models: list[int] = dataclasses.field(default_factory=list)

    tracking: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    """Pipeline stage execution info keyed by stage id made available for
    pipeline orchestration (e.g. policies)"""

    max_timestamp: int = -1

    previous_model_id: int | None = None

    previous_largest_keys: set[int] = dataclasses.field(default_factory=set)

    @property
    def maximum_triggers_reached(self) -> bool:
        return self.maximum_triggers is not None and len(self.triggers) >= self.maximum_triggers


class ConfigLogs(BaseModel):
    system: ModynConfig
    pipeline: ModynPipelineConfig


class StageInfo(BaseModel):
    """Some stages might want to persist additional information in the logs.

    This can be done using the `.info` attribute of the `StageLog` class which stores subtypes of `StageInfo`.
    `StageInfo` class is therefore intended to be subclassed for different pipeline stage information.
    """

    def df_columns(self) -> list[str]:
        """Provide the column names of the DataFrame representation of the
        data."""
        return []

    @property
    def df_row(self) -> tuple:
        """While appending StageLog subclasses to `StageLog.info` is sufficient
        to persist additional information in the logs, this method is used to
        provide a DataFrame representation of the data for analytical purposes.

        Returns:
            The dataframe rows.
        """
        return ()


class FetchDataInfo(StageInfo):
    num_samples: int = Field(..., description="Number of samples processed in the new data.")
    trigger_indexes: list[int] = Field(..., description="Indices of triggers in the new data.")

    def df_columns(self) -> list[str]:
        """Provide the column names of the DataFrame representation of the
        data."""
        return ["num_samples", "trigger_indexes"]

    @override
    @property
    def df_row(self) -> tuple:
        return (self.num_samples, str(self.trigger_indexes))


class ProcessNewDataInfo(StageInfo):
    fetch_time: int = Field(..., description="Time in milliseconds the fetch took")
    num_samples: int = Field(..., description="Number of samples processed")
    trigger_indexes: list[int] = Field(..., description="Indices of triggers in the new data.")

    def df_columns(self) -> list[str]:
        """Provide the column names of the DataFrame representation of the
        data."""
        return ["fetch_time", "num_samples"]

    @override
    @property
    def df_row(self) -> tuple:
        return (self.fetch_time, self.num_samples)


class EvaluateTriggerInfo(StageInfo):
    batch_size: int
    trigger_indexes: list[int] = Field(default_factory=list)
    trigger_eval_times: list[int] = Field(default_factory=list)
    """Time in milliseconds that every next(...) call of the
    trigger.inform(...) generator took."""

    trigger_evaluation_log: TriggerPolicyEvaluationLog = Field(default_factory=TriggerPolicyEvaluationLog)

    def df_columns(self) -> list[str]:
        """Provide the column names of the DataFrame representation of the
        data."""
        return ["batch_size", "trigger_indexes"]

    @override
    @property
    def df_row(self) -> tuple:
        return (self.batch_size, list(self.trigger_indexes))


class _TriggerLogMixin(StageInfo):
    trigger_i: int
    """The index of the trigger in the list of all triggers of this batch."""

    trigger_index: int
    """The index in the dataset where the trigger was initiated."""

    trigger_id: int
    """The identifier of the trigger received from the selector."""


class SelectorInformTriggerInfo(_TriggerLogMixin):
    selector_log: dict[str, Any]
    num_samples_in_trigger: int

    def df_columns(self) -> list[str]:
        """Provide the column names of the DataFrame representation of the
        data."""
        return ["trigger_i", "trigger_index", "trigger_id", "num_samples_in_trigger"]

    @override
    @property
    def df_row(self) -> tuple:
        return (self.trigger_i, self.trigger_index, self.trigger_i, self.num_samples_in_trigger)


class TriggerExecutionInfo(_TriggerLogMixin):
    first_timestamp: int | None
    last_timestamp: int | None

    def df_columns(self) -> list[str]:
        """Provide the column names of the DataFrame representation of the
        data."""
        return ["trigger_i", "trigger_index", "trigger_id", "first_timestamp", "last_timestamp"]

    @override
    @property
    def df_row(self) -> tuple:
        return (self.trigger_i, self.trigger_index, self.trigger_id, self.first_timestamp, self.last_timestamp)


class _TrainInfoMixin(StageInfo):
    trigger_id: int
    training_id: int


class TrainingInfo(_TrainInfoMixin):
    trainer_log: dict[str, Any]

    def df_columns(self) -> list[str]:
        """Provide the column names of the DataFrame representation of the
        data."""
        return ["trigger_id", "training_id", "num_batches", "num_samples", "train_time_at_trainer"]

    @override
    @property
    def df_row(self) -> tuple:
        return (
            self.trigger_id,
            self.training_id,
            self.trainer_log["num_batches"],
            self.trainer_log["num_samples"],
            self.trainer_log["total_train"],
        )


class StoreModelInfo(_TrainInfoMixin):
    id_model: int  # model_ prefix not allowed in pydantic

    def df_columns(self) -> list[str]:
        """Provide the column names of the DataFrame representation of the
        data."""
        return ["trigger_id", "training_id", "id_model"]

    @override
    @property
    def df_row(self) -> tuple:
        return (self.trigger_id, self.training_id, self.id_model)


class SingleEvaluationInfo(StageInfo):
    eval_request: EvalRequest
    results: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None

    def df_columns(self) -> list[str]:
        """Provide the column names of the DataFrame representation of the
        data."""
        return [
            "trigger_id",
            "training_id",
            "id_model",
            "currently_active_model",
            "currently_trained_model",
            "eval_handler",
            "dataset_id",
            "interval_start",
            "interval_end",
            "num_samples",
        ]

    @override
    @property
    def df_row(self) -> tuple:
        return (
            self.eval_request.trigger_id,
            self.eval_request.training_id,
            self.eval_request.id_model,
            self.eval_request.currently_active_model,
            self.eval_request.currently_trained_model,
            self.eval_request.eval_handler,
            self.eval_request.dataset_id,
            self.eval_request.interval_start,
            self.eval_request.interval_end,
            self.results.get("dataset_size", 0),
        )


class MultiEvaluationInfo(StageInfo):
    dataset_id: str
    id_model: int
    interval_results: list[SingleEvaluationInfo] = []

    def df_columns(self) -> list[str]:
        """Provide the column names of the DataFrame representation of the
        data."""
        return ["id_model", "dataset_id", "num_intervals"]

    @override
    @property
    def df_row(self) -> tuple:
        return (self.id_model, self.dataset_id, len(self.interval_results))

    @classmethod
    def results_df(cls, infos: list[MultiEvaluationInfo]) -> pd.DataFrame:
        """As one evaluation can have multiple metrics, we return a DataFrame
        with one row per metric."""
        return pd.DataFrame(
            [
                (
                    # per request
                    single_info.eval_request.trigger_id,
                    single_info.eval_request.training_id,
                    single_info.eval_request.id_model,
                    single_info.eval_request.currently_active_model,
                    single_info.eval_request.currently_trained_model,
                    single_info.eval_request.eval_handler,
                    single_info.eval_request.dataset_id,
                    single_info.eval_request.interval_start,
                    single_info.eval_request.interval_end,
                    single_info.results.get("dataset_size", 0),
                    # per metric
                    metric["name"],
                    metric["result"],
                )
                for multi_info in infos
                for single_info in multi_info.interval_results
                if single_info.failure_reason is None and single_info.eval_request
                for metric in single_info.results["metrics"]  # pylint: disable=unsubscriptable-object
            ],
            columns=[
                "trigger_id",
                "training_id",
                "id_model",
                "currently_active_model",
                "currently_trained_model",
                "eval_handler",
                "dataset_id",
                "interval_start",
                "interval_end",
                "dataset_size",
                "metric",
                "value",
            ],
        )

    @classmethod
    def requests_df(cls, stage_logs: list[StageLog]) -> pd.DataFrame:
        """As one evaluation can have multiple metrics, we return a DataFrame
        with one row per metric."""
        parent_columns: list[str] = []
        if stage_logs:
            parent_columns = stage_logs[0].df_columns(False)
        return pd.DataFrame(
            [
                stage_log.df_row(False)
                + (
                    # per request
                    single_info.eval_request.trigger_id,
                    single_info.eval_request.training_id,
                    single_info.eval_request.id_model,
                    single_info.eval_request.currently_active_model,
                    single_info.eval_request.currently_trained_model,
                    single_info.eval_request.eval_handler,
                    single_info.eval_request.dataset_id,
                    single_info.eval_request.interval_start,
                    single_info.eval_request.interval_end,
                    single_info.results.get("dataset_size", 0),
                )
                for stage_log in stage_logs
                if isinstance(stage_log.info, MultiEvaluationInfo)
                for single_info in stage_log.info.interval_results
                if single_info.failure_reason is None and single_info.eval_request
            ],
            columns=(
                parent_columns
                + [
                    "trigger_id",
                    "training_id",
                    "id_model",
                    "currently_active_model",
                    "currently_trained_model",
                    "eval_handler",
                    "dataset_id",
                    "interval_start",
                    "interval_end",
                    "dataset_size",
                ]
            ),
        )


class SelectorInformInfo(StageInfo):
    selector_log: dict[str, Any] | None
    remaining_data: bool
    trigger_indexes: list[int]

    def df_columns(self) -> list[str]:
        """Provide the column names of the DataFrame representation of the
        data."""
        return ["remaining_data", "trigger_indexes"]

    @override
    @property
    def df_row(self) -> tuple:
        return (self.remaining_data, self.trigger_indexes)


StageInfoUnion = Union[  # noqa: UP007
    FetchDataInfo,
    ProcessNewDataInfo,
    EvaluateTriggerInfo,
    SelectorInformTriggerInfo,
    TriggerExecutionInfo,
    TrainingInfo,
    StoreModelInfo,
    MultiEvaluationInfo,
    SingleEvaluationInfo,
    SelectorInformInfo,
]


class StageLog(BaseModel):
    id: str
    """Identifier for the pipeline stage, PipelineStage.name in most cases."""

    id_seq_num: int
    """Identifies the log within the group of logs with the same id (given by
    PipelineStage).

    Used for aggregation over multiple pipeline runs.
    """

    # experiment time
    start: datetime.datetime
    end: datetime.datetime | None = Field(None)
    """Timestamp when the decorated function exits.

    If decorated functions yields a generator, this will be the time
    when the generator is returned, not when the generator is exhausted.
    """

    duration: datetime.timedelta | None = Field(None)
    """As pipeline stages can be lazily evaluated generators where other
    computation steps are interleaved, `end-start` is not always the actual
    duration this stage spent in computing."""

    # dataset time of last seen sample
    batch_idx: int
    sample_idx: int
    sample_time: int
    trigger_idx: int

    # stage specific log info
    info: StageInfo | None = Field(None)

    def df_columns(self, extended: bool = False) -> list[str]:
        """Provide the column names of the DataFrame representation of the
        data."""
        return ["id", "start", "end", "duration", "batch_idx", "sample_idx", "sample_time", "trigger_idx"] + (
            self.info.df_columns() if extended and self.info else []
        )

    def df_row(self, extended: bool = False) -> tuple:
        return (
            self.id,
            self.start,
            self.end,
            self.duration,
            self.batch_idx,
            self.sample_idx,
            self.sample_time,
            self.trigger_idx,
        ) + (self.info.df_row if extended and self.info else ())

    # (De)Serialization to enable parsing all classes in the StageInfoUnion;
    # with that logic we avoid having to add disciminator fields to every subclass of StageInfo

    @model_serializer(mode="wrap")
    def serialize(self, handler: Callable[..., dict[str, Any]]) -> dict[str, Any]:
        default_serializer = handler(self)
        return {
            **default_serializer,
            "info": self.info.model_dump(by_alias=True) if self.info else None,
            **({"info_type": self.info.__class__.__name__} if self.info else {}),
        }

    @model_validator(mode="before")
    @classmethod
    def deserializer(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("info_type") and data.get("info"):
                info_type = cast(type[StageInfo], globals().get(data["info_type"]))
                data["info"] = info_type.model_validate(data["info"])
            else:
                data["info"] = None
        return data

    @classmethod
    def df(cls, stage_logs: Iterator[StageLog], extended: bool = False) -> pd.DataFrame:
        """Provides a DataFrame with the log information of this stage.

        To conveniently allow analysis of lists of log entries, this method provides a DataFrame representation of the
        log entry.

        Args:
            extended: If True, include the columns of the info attribute. Requires all logs to have the same type.

        Returns:
            A DataFrame row with the log information of this stage.
        """
        if not stage_logs:
            return pd.DataFrame()
        stage_logs, iter_copy = itertools.tee(stage_logs)
        return pd.DataFrame(
            [stage.df_row(extended=extended) for stage in stage_logs],
            columns=next(iter_copy).df_columns(extended=extended),
        )


class SupervisorLogs(BaseModel):
    stage_runs: list[StageLog] = Field(default_factory=list)

    def clear(self) -> None:
        self.stage_runs.clear()

    def merge(self, logs: list[StageLog]) -> SupervisorLogs:
        self.stage_runs = self.stage_runs + logs
        return self

    @property
    def df(self) -> pd.DataFrame:
        """Provides a dataframe with log information of all stages which helps
        to easily plot pipeline run metrics.

        Returns:
            A DataFrame with the core log information of all stages
            (not including the additional info property of the StageLogs).
        """

        return pd.DataFrame(
            data=[
                (
                    stage.id,
                    stage.start,
                    stage.end,
                    stage.duration,
                    stage.batch_idx,
                    stage.sample_idx,
                    stage.sample_time,
                    stage.trigger_idx,
                )
                for stage in self.stage_runs
            ],
            columns=[
                "id",
                "start",
                "end",
                "duration",
                "batch_idx",
                "sample_idx",
                "sample_time",
                "trigger_idx",
            ],
        )


class PipelineLogs(BaseModel):
    """Wrapped logs for the pipeline executor.

    This file maintains the log directory:
    logs
    ├── pipeline_1
    │   ├── pipeline.log (final logfile being made available at the end of the pipeline execution)
    │   ├── pipeline.part.log (initial logfile including static fields)
    │   ├── supervisor.part_0.log (incremental log files)
    │   ├── supervisor.part_1.log
    │   └── supervisor.part_2.log
    └── pipeline_2
    """

    # static logs

    pipeline_id: int
    commit_sha: str | None = Field(None, description="The commit SHA that the pipeline was executed on.")
    pipeline_stages: dict[str, tuple[int, list[str]]]
    """List of all pipeline stages, their execution order index, and their
    parent stages."""

    config: ConfigLogs

    experiment: bool
    start_replay_at: int | None = Field(None, description="Epoch to start replaying at")
    stop_replay_at: int | None = Field(None, description="Epoch to stop replaying at")

    # incremental logs

    supervisor_logs: SupervisorLogs = Field(default_factory=SupervisorLogs)

    # metadata
    partial_idx: int = Field(0)

    def materialize(self, log_dir_path: Path, mode: Literal["initial", "increment", "final"] = "increment") -> None:
        """Materialize the logs to log files.

        If run with pytest, log_file_path and mode will be ignored.

        Args:
            log_dir_path: The path to the logging directory.
            mode: The mode to materialize the logs. Initial will output static fields, increment will output only
                new items since the last materialization, and final will merge all logs.
        """
        if "PYTEST_CURRENT_TEST" in os.environ:
            self.model_dump_json(by_alias=True)  # Enforce serialization to catch issues
            return  # But don't actually store in tests

        # ensure empty log directory
        pipeline_logdir = log_dir_path / f"pipeline_{self.pipeline_id}"

        if mode == "initial" and pipeline_logdir.exists():
            # if not empty move the previous content to .backup_timestamp directory
            backup_dir = log_dir_path / f"pipeline_{self.pipeline_id}.backup_{datetime.datetime.now().isoformat()}"
            pipeline_logdir.rename(backup_dir)

        pipeline_logdir.mkdir(parents=True, exist_ok=True)
        (pipeline_logdir / ".name").write_text(self.config.pipeline.pipeline.name)

        # output logs

        if mode == "initial":
            try:
                self.commit_sha = get_head_sha()
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(f"Failed to get commit SHA: {e}")

            with open(pipeline_logdir / "pipeline.part.log", "w", encoding="utf-8") as logfile:
                logfile.write(self.model_dump_json(indent=2, exclude={"supervisor", "partial_idx"}, by_alias=True))
            return

        if mode == "increment":
            with open(pipeline_logdir / f"supervisor_part_{self.partial_idx}.log", "w", encoding="utf-8") as logfile:
                logfile.write(self.supervisor_logs.model_dump_json(by_alias=True, indent=2))

            self.supervisor_logs.clear()
            self.partial_idx += 1
            return

        # final: merge increment logs
        supervisor_files = sorted(pipeline_logdir.glob("supervisor_part_*.log"))
        partial_supervisor_logs = [
            SupervisorLogs.model_validate_json(file.read_text(encoding="utf-8")) for file in supervisor_files
        ]
        self.supervisor_logs = self.supervisor_logs.merge(
            [stage_log for sv_logs in partial_supervisor_logs for stage_log in sv_logs.stage_runs]
        )

        with open(pipeline_logdir / "pipeline.log", "w", encoding="utf-8") as logfile:
            logfile.write(self.model_dump_json(by_alias=True, indent=2))

        self.supervisor_logs.clear()
