# pylint: disable=E1133

from __future__ import annotations

import dataclasses
import datetime
import multiprocessing as mp
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import pandas as pd
from modyn.config.schema.config import ModynConfig
from modyn.config.schema.pipeline import ModynPipelineConfig
from modyn.supervisor.internal.grpc.enums import PipelineStage
from modyn.supervisor.internal.utils.evaluation_status_reporter import EvaluationStatusReporter
from pydantic import BaseModel, Field, model_validator
from typing_extensions import override


@dataclass
class PipelineOptions:
    """
    Wrapped cli argument bundle for the pipeline executor.
    """

    start_timestamp: int
    pipeline_id: int
    modyn_config: ModynConfig
    pipeline_config: ModynPipelineConfig
    eval_directory: Path
    supervisor_supported_eval_result_writers: dict
    exception_queue: mp.Queue
    pipeline_status_queue: mp.Queue
    training_status_queue: mp.Queue
    eval_status_queue: mp.Queue
    start_replay_at: int | None = None
    stop_replay_at: int | None = None
    maximum_triggers: int | None = None

    @property
    def log_directory(self) -> Path:
        return self.eval_directory

    @property
    def experiment_mode(self) -> bool:
        return self.start_replay_at is not None

    @cached_property
    def selector_batch_size(self) -> int:
        dataset_id = self.pipeline_config.data.dataset_id

        for dataset in self.modyn_config.storage.datasets:
            if dataset.name == dataset_id:
                return dataset.selector_batch_size

        assert False, f"Dataset with id {dataset_id} not found in storage configuration."

    @model_validator(mode="after")
    def validate_replay(self) -> PipelineOptions:
        if self.start_replay_at is None and self.stop_replay_at is not None:
            raise ValueError("`stop_replay_at` can only be used in conjunction with `start_replay_at`.")
        return self


@dataclass
class PipelineBatchState:
    """Pipeline artifacts shared across stages during the processing of one batch."""

    data: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)
    remaining_data: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)

    triggering_indices: list[int] = dataclasses.field(default_factory=list)
    previous_trigger_idx: int = 0

    trigger_index: int = -1
    """The index in the dataset where the trigger was initiated."""

    trigger_id: int = -1
    """The identifier of the trigger received from the selector."""

    evaluations: dict[int, EvaluationStatusReporter] = dataclasses.field(default_factory=dict)


@dataclass
class NewDataState:
    """Pipeline artifacts shared across stages during the processing of new data."""

    # PipelineStage.REPLAY_DATA, PipelineStage.FETCH_NEW_DATA
    data: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)
    """Stores the new unprocessed data to be processed in `HANDLE_NEW_DATA`."""

    fetch_time: int = -1
    """milliseconds the last data fetch took"""

    # PipelineStage.REPLAY_DATA

    max_timestamp: int = -1

    # PipelineStage.FETCH_NEW_DATA

    had_trigger: bool = False
    """Whether the current new data arrival caused at least one trigger."""


@dataclass
class ExecutionState(PipelineOptions):
    """Represent the state of the pipeline executor including artifacts of pipeline stages."""

    current_sample_idx: int = Field(0)
    current_sample_time: int = Field(0)
    """The unix timestamp of the last sample seen by the pipeline executor."""

    num_triggers: int = Field(0)
    trained_models: list[int] = dataclasses.field(default_factory=list)
    triggers: list[int] = dataclasses.field(default_factory=list)

    tracking: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    """Pipeline stage execution info keyed by stage id made available for pipeline orchestration (e.g. policies)"""

    # Internal state for pipeline steps

    # RUN_TRAINING, STORE_TRAINED_MODEL
    previous_model_id: Optional[int] = None
    training_id: Optional[int] = None

    new_data: NewDataState = dataclasses.field(default_factory=NewDataState)
    previous_largest_keys: set[int] = dataclasses.field(default_factory=set)

    # new_data_batch_pipeline
    batch: PipelineBatchState = dataclasses.field(default_factory=PipelineBatchState)

    # this is to store the first and last timestamp of the remaining data after handling all triggers
    remaining_data_range: Optional[tuple[int, int]] = None

    # INFORM_SELECTOR_AND_TRIGGER
    previous_batch_had_trigger: bool = False


@dataclass
class RegisteredStage:
    """Represent a registered pipeline stage that includes a callable function and the next stage."""

    stage: PipelineStage
    func: Callable
    next: PipelineStage | None = None
    """If next stage if None, the next stage will be decided by the return value of the current stage's `func`.

    If both are None, the pipeline ends.
    """

    log: bool = True
    track: bool = False


class ConfigLogs(BaseModel):
    system: ModynConfig
    pipeline: ModynPipelineConfig


class StageInfo(BaseModel):
    """Base class for stage log info. Has no members, intended to be subclassed"""

    def online_df(self) -> pd.DataFrame | None:
        """
        Some stages might want to store data in the online DataFrame.

        This data can be used by pipeline steps such as trigger policies.

        Returns:
            A DataFrame if the stage should collect data, else None.
        """
        return None


class HandleNewDataInfo(StageInfo):
    fetch_time: int = Field(..., description="Time in milliseconds the fetch took")
    num_samples: int = Field(..., description="Number of samples processed")
    had_trigger: bool

    @override
    def online_df(self) -> pd.DataFrame | None:
        return pd.DataFrame([self.fetch_time, self.num_samples], columns=["fetch_time", "num_samples"])


class EvaluateTriggerInfo(StageInfo):
    batch_size: int
    num_triggers: int

    @override
    def online_df(self) -> pd.DataFrame | None:
        return pd.DataFrame([self.batch_size, self.num_triggers], columns=["batch_size", "num_triggers"])


class _TriggerLogMixin(StageInfo):
    trigger_index: int
    """The index in the dataset where the trigger was initiated."""

    trigger_id: int
    """The identifier of the trigger received from the selector."""


class SelectorInformTriggerLog(_TriggerLogMixin):
    selector_log: dict[str, Any]

    first_timestamp: int
    last_timestamp: int

    @override
    def online_df(self) -> pd.DataFrame | None:
        return pd.DataFrame([self.trigger_index, self.trigger_id], columns=["trigger_index", "trigger_id"])


class _TrainLogMixin(_TriggerLogMixin):
    training_id: int


class TrainingInfo(_TrainLogMixin):
    trainer_log: dict[str, Any]

    @override
    def online_df(self) -> pd.DataFrame | None:
        return pd.DataFrame(
            [self.trigger_index, self.trigger_id, self.training_id],
            columns=["trigger_index", "trigger_id", "training_id"],
        )


class StoreModelInfo(_TrainLogMixin):
    model_id: int

    @override
    def online_df(self) -> pd.DataFrame | None:
        return pd.DataFrame(
            [self.trigger_index, self.trigger_id, self.training_id, self.model_id],
            columns=["trigger_index", "trigger_id", "training_id", "model_id"],
        )


class EvaluationInfo(StoreModelInfo):
    pass


class SelectorInformLog(StageInfo):
    selector_log: dict[str, Any]
    seen_trigger: bool

    @override
    def online_df(self) -> pd.DataFrame | None:
        return pd.DataFrame([self.seen_trigger], columns=["seen_trigger"])


class StageLog(BaseModel):
    id: str
    """Identifier for the pipeline stage, PipelineStage.name in most cases"""

    # experiment time
    start: datetime.datetime
    end: datetime.datetime | None = Field(None)

    # dataset time of last seen sample
    sample_idx: int
    sample_time: int

    # stage specific log info
    info: StageInfo | None = Field(None)

    def online_df(self, extended: bool = False) -> pd.DataFrame | None:
        """Args:
        extended: If True, include the columns of the info attribute. Requires all logs to have the same type.
        """
        df = pd.DataFrame(
            [self.id, self.duration, self.sample_time], columns=["id", "duration", "sample_idx", "sample_time"]
        )
        info_df = self.info.online_df() if self.info else None
        if info_df and extended:
            # add additional columns
            df = pd.concat([df, info_df], axis=1)

        return df

    @property
    def duration(self) -> datetime.timedelta:
        assert self.end is not None
        return self.end - self.start


class SupervisorLogs(BaseModel):
    stage_runs: list[StageLog] = Field(default_factory=list)

    def clear(self) -> None:
        self.stage_runs.clear()

    def merge(self, logs: list[StageLog]) -> SupervisorLogs:
        self.stage_runs = self.stage_runs + logs
        return self

    @property
    def stage_runs_df(self) -> pd.DataFrame:
        return pd.DataFrame([stage_run.df_row() for stage_run in self.stage_runs], columns=StageLog.df_columns())


class PipelineLogs(BaseModel):
    """
    Wrapped logs for the pipeline executor.

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

    config: ConfigLogs

    experiment: bool
    start_replay_at: int | None = Field(None, description="Epoch to start replaying at")
    stop_replay_at: int | None = Field(None, description="Epoch to stop replaying at")

    # incremental logs

    supervisor: SupervisorLogs = Field(default_factory=SupervisorLogs)

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
            self.model_dump_json()  # Enforce serialization to catch issues
            return  # But don't actually store in tests

        # ensure empty log directory
        pipeline_logdir = log_dir_path / f"pipeline_{self.pipeline_id}"

        if pipeline_logdir.exists():
            # if not empty move the previous content to .backup_timetamp directory
            backup_dir = log_dir_path / f"pipeline_{self.pipeline_id}.backup_{datetime.datetime.now().isoformat()}"
            pipeline_logdir.rename(backup_dir)

        pipeline_logdir.mkdir(parents=True, exist_ok=True)

        # output logs

        if mode == "initial":
            with open(pipeline_logdir / "pipeline.part.log", "w", encoding="utf-8") as logfile:
                logfile.write(self.model_dump_json(indent=2, exclude={"supervisor", "partial_idx"}, by_alias=True))
            return

        if mode == "increment":
            with open(pipeline_logdir / f"supervisor_part_{self.partial_idx}.log", "w", encoding="utf-8") as logfile:
                logfile.write(self.supervisor.model_dump(indent=2))

            self.supervisor.clear()
            self.partial_idx += 1
            return

        # final: merge increment logs
        supervisor_files = sorted(pipeline_logdir.glob(f"pipeline_{self.pipeline_id}/supervisor.part_*.log"))
        partial_supervisor_logs = [
            SupervisorLogs.model_validate_json(file.read_text(encoding="utf-8")) for file in supervisor_files
        ]
        self.supervisor = self.supervisor.merge(partial_supervisor_logs)

        with open(pipeline_logdir / "pipeline.log", "w", encoding="utf-8") as logfile:
            logfile.write(self.model_dump_json(indent=2))

        self.supervisor.clear()
