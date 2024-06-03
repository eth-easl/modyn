"""
Evaluation configurations are similar to the configuration of a training. The same questions need to be answered:
- What data should be used?
- When should the evaluation be triggered?

Theoretically, we could reuse models like the trivial trigger configs (TimeTriggerConfig, DataAmountTriggerConfig).
However, as some trigger policies like data drift triggers don't make sense and training triggers are part of the
core pipeline logic with potential additional configurations, we decided to create a models here.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union, cast, get_args

from modyn.config.schema.modyn_base_model import ModynBaseModel
from modyn.config.schema.pipeline.base import REGEX_TIME_UNIT
from modyn.config.schema.pipeline.data import DataConfig
from modyn.utils import validate_timestr
from modyn.utils.utils import SECONDS_PER_UNIT
from pydantic import Field, NonNegativeInt, field_validator, model_validator
from pyparsing import cached_property
from typing_extensions import Self

# -------------------------------------------------------------------------------------------------------------------- #
#                                               EvalHandler configuration                                              #
# -------------------------------------------------------------------------------------------------------------------- #

# Evaluation metric


class Metric(ModynBaseModel):
    name: str = Field(description="The name of the evaluation metric.")
    config: Optional[Dict[str, Any]] = Field(None, description="Configuration for the evaluation metric.")
    evaluation_transformer_function: Optional[str] = Field(
        None,
        description="A function used to transform the model output before evaluation.",
    )


# Evaluation strategy: how to evaluate the model (what data to use, how to split it, etc.)


class MatrixEvalStrategyConfig(ModynBaseModel):
    type: Literal["MatrixEvalStrategy"] = Field("MatrixEvalStrategy")
    eval_every: str = Field(
        description="The interval length for the evaluation "
        "specified by an integer followed by a time unit (e.g. '100s')."
    )
    eval_start_from: NonNegativeInt = Field(
        description="The timestamp from which the evaluation should start (inclusive). This timestamp is in seconds."
    )
    eval_end_at: NonNegativeInt = Field(
        description="The timestamp at which the evaluation should end (exclusive). This timestamp is in seconds."
    )

    @field_validator("eval_every")
    @classmethod
    def validate_eval_every(cls, value: str) -> str:
        if not validate_timestr(value):
            raise ValueError("eval_every must be a valid time string")
        return value

    @model_validator(mode="after")
    def eval_end_at_must_be_larger(self) -> Self:
        if self.eval_start_from >= self.eval_end_at:
            raise ValueError("eval_end_at must be larger than eval_start_from")
        return self


_EVAL_INTERVAL_BOUND_PATTERN = rf"[+-]?\s*((\d+\s*({REGEX_TIME_UNIT}))|(inf))"
_EVAL_INTERVAL_PATTERN = (
    rf"^(\[|\()\s*({_EVAL_INTERVAL_BOUND_PATTERN})\s*(,|;)\s*({_EVAL_INTERVAL_BOUND_PATTERN})\s*(\)|\])$"
)


TimeUnit = Literal["s", "m", "h", "d", "w", "y"]


class IntervalEvalStrategyConfig(ModynBaseModel):
    """Allows to evaluate a model on an interval that is centered around the time of the evaluation trigger.

    Note: when specifying the left and right offsets you can choose between several
        time units: 's', 'm', 'h', 'd', 'w', 'y'

    Note: for the following we define `training_interval` as the interval that is used for training the model.

    The position of the offset (bounds inclusive):
    - numbers > 0: the boundary of the interval is the end_training_interval + offset
    - numbers < 0: the boundary of the interval is the start_training_interval - offset
    - 'inf': the boundary of the interval is the dataset end
    - '-inf': the boundary of the interval is the dataset start
    - '0', '-0', '+0': zero refers to the training_interval.
        - "-0" = start of the training_interval (start_training_interval)
        - "+0" = end of the training_interval (end_training_interval)
        - '0': start_training_interval if used as left offset, end_training_interval if used as right offset

    Checkout `docs/EVALUATION.md` for a graphical representation of the options.
    """

    type: Literal["IntervalEvalStrategy"] = Field("IntervalEvalStrategy")
    interval: str = Field(
        description="The two-sided interval specified by two offsets relative to the training interval.",
        pattern=_EVAL_INTERVAL_PATTERN,
    )
    """e.g. [-2d; +0y], [-inf; +inf]=(-inf; +inf)"""

    @cached_property
    def left(self) -> str:
        return self._parsed[0]

    @cached_property
    def left_unit(self) -> TimeUnit:
        return self._parsed[1]

    @cached_property
    def left_bound_inclusive(self) -> bool:
        return self._parsed[2]

    @cached_property
    def right(self) -> str:
        return self._parsed[3]

    @cached_property
    def right_unit(self) -> TimeUnit:
        return self._parsed[4]

    @cached_property
    def right_bound_inclusive(self) -> bool:
        return self._parsed[5]

    @field_validator("interval")
    @classmethod
    def clean_interval(cls, value: str) -> str:
        return value.strip().replace(" ", "").replace(",", ";")

    @staticmethod
    def _bounds_type(offset: str) -> int:  # pylint: disable=too-many-return-statements
        """Assign temporal sort keys for the different bounds."""
        if offset == "-inf":
            return -3
        if offset == "-0":
            return -1
        if offset == "0":
            return 0
        if offset.startswith("-"):
            return -2
        if offset == "+0":
            return 1
        if offset == "+inf":
            return 3
        return 2

    @model_validator(mode="after")
    def check_offsets(self) -> IntervalEvalStrategyConfig:
        # sequential list of offsets: -inf, -number, -0, 0, +0, +number, +inf
        if IntervalEvalStrategyConfig._bounds_type(self.left) > IntervalEvalStrategyConfig._bounds_type(self.right):
            raise ValueError("The left offset must be smaller than the right offset.")
        return self

    @cached_property
    def _parsed(self) -> tuple[str, TimeUnit, bool, str, TimeUnit, bool]:
        """Returns:
        the bounds offsets of the interval as seconds
        0: left bound offset
        1: left unit
        2: left_bound_inclusive
        3: right bound offset
        4: right unit
        5: right_bound_inclusive
        """
        interval = str(self.interval)
        left_bound_inclusive = interval[0] == "["
        right_bound_inclusive = interval[-1] == "]"
        interval = interval[1:-1]
        left_raw, right_raw = interval.split(";")
        left_is_inf = "inf" in left_raw
        right_is_inf = "inf" in right_raw

        if left_is_inf:
            assert "+" not in left_raw, "Left bound of the interval cannot be +inf."
            left = "-inf"
            left_unit = "d"
        else:
            left_unit = left_raw[-1]
            left = left_raw[:-1]

        if right_is_inf:
            assert "-" not in right_raw, "Right bound of the interval cannot be -inf."
            right = "+inf"
            right_unit = "d"
        else:
            right_unit = right_raw[-1]
            right = right_raw[:-1]

        assert left_unit in get_args(TimeUnit)
        assert right_unit in get_args(TimeUnit)
        return (
            left,
            cast(TimeUnit, left_unit),
            left_bound_inclusive,
            right,
            cast(TimeUnit, right_unit),
            right_bound_inclusive,
        )


class UntilNextTriggerEvalStrategyConfig(ModynBaseModel):
    """This evaluation strategy will evaluate the model on the intervals between two consecutive triggers.

    This exactly reflects the time span where one model is used for inference.
    """

    type: Literal["UntilNextTriggerEvalStrategy"] = Field("UntilNextTriggerEvalStrategy")


EvalStrategyConfig = Annotated[
    Union[MatrixEvalStrategyConfig, IntervalEvalStrategyConfig, UntilNextTriggerEvalStrategyConfig],
    Field(discriminator="type"),
]


# Evaluation trigger config: when to invoke triggering


class _MatrixEvalTriggerConfig(ModynBaseModel):
    """Base class for EvalalTriggerConfigs that allow to evaluate every model to be evaluated at the trigger times"""

    matrix: bool = Field(
        False, description="Weather to evaluate all models at this these points or always only the most recent one"
    )


class StaticEvalTriggerConfig(_MatrixEvalTriggerConfig):
    """A user defined sequence of timestamps or sample indexes at which the evaluation should be performed.

    Note: This strategy will run evaluations after the core pipeline.
    """

    mode: Literal["static"] = Field("static")
    at: set[int] = Field(
        description="List of timestamps or sample indexes at which the evaluation should be performed."
    )
    unit: Literal["epoch", "sample_index"]
    """Whether to trigger at specific timestamps given or after a specific amount of seen samples.

    sample indexes refer to the training set of the pipeline.
    """

    start_timestamp: int = Field(0, description="The timestamp at which evaluations are started to be checked.")


class PeriodicEvalTriggerConfig(_MatrixEvalTriggerConfig):
    """Comparable to TimeTriggerConfig and DataAmountTriggerConfig, but for scheduling evaluation runs.

    Note: This strategy will run evaluations after the core pipeline.
    """

    mode: Literal["periodic"] = Field("periodic")
    every: str = Field(
        description=(
            "Interval length for the evaluation as an integer followed by a time unit (s, m, h, d, w, y) "
            "or nothing (#samples)"
        ),
        pattern=rf"^\d+({REGEX_TIME_UNIT})?$",
    )
    start_timestamp: int | None = Field(
        None, description="The timestamp at which evaluations are started to be checked."
    )
    end_timestamp: int | None = Field(
        None, description="The timestamp at which evaluations are stopped to be checked. Needed iff unit is 'epoch'."
    )

    @property
    def unit_type(self) -> Literal["samples", "epoch"]:
        return "samples" if self.every.isnumeric() else "epoch"

    @cached_property
    def every_sec_or_samples(self) -> int:
        if self.unit_type == "samples":
            return int(self.every)
        unit = str(self.every)[-1:]
        num = int(str(self.every)[:-1])
        return num * SECONDS_PER_UNIT[unit]

    @model_validator(mode="after")
    def check_start_end_timestamp(self) -> PeriodicEvalTriggerConfig:
        is_ok = (self.unit_type == "epoch") == (self.start_timestamp is not None) == (self.end_timestamp is not None)
        if not is_ok:
            raise ValueError("start_timestamp and end_timestamp must be set iff unit is 'epoch'")
        return self


class AfterTrainingEvalTriggerConfig(ModynBaseModel):
    mode: Literal["after_training"] = Field("after_training")


EvalTriggerConfig = Annotated[
    Union[StaticEvalTriggerConfig, PeriodicEvalTriggerConfig, AfterTrainingEvalTriggerConfig],
    Field(discriminator="mode"),
]


class EvalHandlerConfig(ModynBaseModel):
    name: str = Field(
        "Modyn EvalHandler", description="The name of the evaluation handler used to identify its outputs in the log"
    )
    strategy: EvalStrategyConfig = Field(description="Defining the strategy and data range to be evaluated on.")
    trigger: EvalTriggerConfig = Field(None, description="Configures when the evaluation should be performed.")
    datasets: List[str] = Field(
        description="All datasets on which the model is evaluated.",
        min_length=1,
    )
    """ Note: the datasets have to be defined in the root EvaluationConfig model."""

    @model_validator(mode="after")
    def check_trigger(self) -> Self:
        if isinstance(self.strategy, UntilNextTriggerEvalStrategyConfig) and not (
            isinstance(self.trigger, AfterTrainingEvalTriggerConfig)
        ):
            raise ValueError("UntilNextTriggerEvalStrategyConfig can only be used with AfterTrainingEvalTriggerConfig.")
        return self


# -------------------------------------------------------------------------------------------------------------------- #
#                                                      Data Config                                                     #
# -------------------------------------------------------------------------------------------------------------------- #


class EvalDataConfig(DataConfig):
    batch_size: int = Field(description="The batch size to be used during evaluation.", ge=1)
    dataloader_workers: int = Field(
        description="The number of data loader workers on the evaluation node that fetch data from storage.", ge=1
    )
    metrics: List[Metric] = Field(
        description="All metrics used to evaluate the model on the given dataset.",
        min_length=1,
    )
    tokenizer: Optional[str] = Field(
        None,
        description="Function to tokenize the input. Must be a class in modyn.models.tokenizers.",
    )


# -------------------------------------------------------------------------------------------------------------------- #
#                                                          IO                                                          #
# -------------------------------------------------------------------------------------------------------------------- #


class ResultWriter(ModynBaseModel):
    name: str = Field(description="The name of the result writer.")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional configuration for the result writer.")


ResultWriterType = Literal["json", "json_dedicated", "tensorboard"]
"""
- json: appends the evaluations to the standard json logfile.
- json_dedicated: dumps the results into dedicated json files for each evaluation.
- tensorboard: output the evaluation to dedicated tensorboard files."""


# -------------------------------------------------------------------------------------------------------------------- #
#                                                      root model                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


class EvaluationConfig(ModynBaseModel):
    handlers: list[EvalHandlerConfig] = Field(
        description="An array of all evaluation handlers that should be used to evaluate the model.",
        min_length=1,
    )
    device: str = Field(description="The device the model should be put on.")
    result_writers: List[ResultWriterType] = Field(
        ["json"],
        description=(
            "List of names that specify in which formats to store the evaluation results. We currently support "
            "json and tensorboard."
        ),
        min_length=1,
    )
    datasets: dict[str, EvalDataConfig] = Field(
        description="An array of all datasets on which the model is evaluated keyed by an arbitrary reference tag.",
        min_length=1,
    )

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, value: dict[str, EvalDataConfig]) -> dict[str, EvalDataConfig]:
        dataset_ids = [dataset.dataset_id for dataset in value.values()]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("Dataset IDs must be unique.")
        return value
