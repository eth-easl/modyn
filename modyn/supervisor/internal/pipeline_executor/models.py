# pylint: disable=E1133

from __future__ import annotations

import dataclasses
import datetime
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from modyn.config.schema.config import ModynConfig
from modyn.config.schema.pipeline import ModynPipelineConfig
from modyn.supervisor.internal.grpc.enums import PipelineStage
from modyn.supervisor.internal.utils.evaluation_status_reporter import EvaluationStatusReporter
from pydantic import BaseModel, Field, model_validator
from typing_extensions import override

logger = logging.getLogger(__name__)


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
    def validate_replay(self) -> PipelineOptions:
        if self.start_replay_at is None and self.stop_replay_at is not None:
            raise ValueError("`stop_replay_at` can only be used in conjunction with `start_replay_at`.")
        return self


@dataclass
class PipelineBatchState:
    """Pipeline artifacts shared across stages during the processing of one batch."""

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
class ExecutionState(PipelineOptions):
    """Represent the state of the pipeline executor including artifacts of pipeline stages."""

    stage: PipelineStage = PipelineStage.INIT
    """The current stage of the pipeline executor."""

    current_sample_index: int = 0
    current_sample_time: int = 0  # unix timestamp
    """The unix timestamp of the last sample seen and processed by the pipeline executor."""

    # this is to store the first and last timestamp of the remaining data after handling all triggers
    remaining_data: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)
    remaining_data_range: Optional[tuple[int, int]] = None

    triggers: list[int] = dataclasses.field(default_factory=list)
    current_training_id: int | None = None
    trained_models: list[int] = dataclasses.field(default_factory=list)

    tracking: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    """Pipeline stage execution info keyed by stage id made available for pipeline orchestration (e.g. policies)"""

    max_timestamp: int = -1

    previous_model_id: Optional[int] = None

    previous_largest_keys: set[int] = dataclasses.field(default_factory=set)

    @property
    def maximum_triggers_reached(self) -> bool:
        return self.maximum_triggers is not None and len(self.triggers) >= self.maximum_triggers


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


class FetchDataInfo(StageInfo):
    num_samples: int = Field(..., description="Number of samples processed in the new data.")
    trigger_indexes: list[int] = Field(..., description="Indices of triggers in the new data.")

    @override
    def online_df(self) -> pd.DataFrame | None:
        return pd.DataFrame([(self.num_samples, str(self.trigger_indexes))], columns=["num_samples", "trigger_indexes"])


class ProcessNewDataInfo(StageInfo):
    fetch_time: int = Field(..., description="Time in milliseconds the fetch took")
    num_samples: int = Field(..., description="Number of samples processed")
    trigger_indexes: list[int] = Field(..., description="Indices of triggers in the new data.")

    @override
    def online_df(self) -> pd.DataFrame | None:
        return pd.DataFrame([(self.fetch_time, self.num_samples)], columns=["fetch_time", "num_samples"])


class EvaluateTriggerInfo(StageInfo):
    batch_size: int
    trigger_indexes: list[int] = Field(default_factory=list)
    trigger_eval_times: list[int] = Field(default_factory=list)
    """Time in milliseconds that every next(...) call of the trigger.inform(...) generator took."""

    @override
    def online_df(self) -> pd.DataFrame | None:
        return pd.DataFrame([(self.batch_size, list(self.trigger_indexes))], columns=["batch_size", "trigger_indexes"])


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

    @override
    def online_df(self) -> pd.DataFrame | None:
        return pd.DataFrame(
            [(self.trigger_i, self.trigger_index, self.trigger_i, self.num_samples_in_trigger)],
            columns=["trigger_i", "trigger_index", "trigger_id", "num_samples_in_trigger"],
        )


class TriggerExecutionInfo(_TriggerLogMixin):
    first_timestamp: int | None
    last_timestamp: int | None

    @override
    def online_df(self) -> pd.DataFrame | None:
        return pd.DataFrame(
            [(self.trigger_i, self.trigger_index, self.trigger_id, self.first_timestamp, self.last_timestamp)],
            columns=["trigger_i", "trigger_index", "trigger_id", "first_timestamp", "last_timestamp"],
        )


class _TrainInfoMixin(StageInfo):
    trigger_id: int
    training_id: int


class TrainingInfo(_TrainInfoMixin):
    trainer_log: dict[str, Any]

    @override
    def online_df(self) -> pd.DataFrame | None:
        return pd.DataFrame([(self.trigger_id, self.training_id)], columns=["trigger_id", "training_id"])


class StoreModelInfo(_TrainInfoMixin):
    id_model: int  # model_ prefix not allowed in pydantic

    @override
    def online_df(self) -> pd.DataFrame | None:
        return pd.DataFrame(
            [(self.trigger_id, self.training_id, self.id_model)],
            columns=["trigger_id", "training_id", "id_model"],
        )


class EvaluationInfo(StoreModelInfo):
    pass


class SelectorInformLog(StageInfo):
    selector_log: dict[str, Any] | None
    remaining_data: bool
    trigger_indexes: list[int]

    @override
    def online_df(self) -> pd.DataFrame | None:
        return pd.DataFrame(
            [(self.remaining_data, self.trigger_indexes)], columns=["remaining_data", "trigger_indexes"]
        )


class StageLog(BaseModel):
    id: str
    """Identifier for the pipeline stage, PipelineStage.name in most cases"""

    # experiment time
    start: datetime.datetime
    end: datetime.datetime | None = Field(None)
    """Timestamp when the decorated function exits. If decorated functions yields a generator, this will be the time
    when the generator is returned, not when the generator is exhausted."""

    duration: datetime.timedelta | None = Field(None)
    """As pipeline stages can be lazily evaluated generators where other computation steps are interleaved,
    `end-start` is not always the actual duration this stage spent in computing."""

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
            [(self.id, self.start, self.end, self.duration, self.sample_idx, self.sample_time)],
            columns=["id", "start", "end", "duration", "sample_idx", "sample_time"],
        )
        info_df = self.info.online_df() if self.info else None
        if info_df is not None and extended:
            # add additional columns
            df = pd.concat([df, info_df], axis=1)

        return df


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
