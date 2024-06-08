from __future__ import annotations

from typing import List, Literal

from modyn.config.schema.base_model import ModynBaseModel
from pydantic import Field

from .strategy import EvalStrategyConfig

EvalHandlerExecutionTime = Literal["after_training"]
"""
after_training: will evaluate the models on the datasets after the training has finished
after_pipeline: will evaluate the models on the datasets after the pipeline has finished
manual: will skip the evaluations in the pipeline run; useful if someone wants to start them with a different
    entry point later on in isolation (useful for debugging and when evaluation is not part of the performance critical
    part of the pipeline, can be executed while the system is busy with other things e.g.)
async_during_pipeline (unimplemented): useful when the pipeline should perform evaluations asynchronously during the
    pipeline (e.g. for continuously generating data for training trigger decisions)
"""


class EvalHandlerConfig(ModynBaseModel):
    """Configuration of one certain evaluation sequence on modyn pipeline execution level.

    Configures what datasets a certain evaluation series should run on, what intervals and models to combine...
    This basically reduces the choices in the space `Datasets x Models x Intervals x TimeOfExecutionInPipeline`
    """

    name: str = Field(
        "Modyn EvalHandler", description="The name of the evaluation handler used to identify its outputs in the log"
    )

    execution_time: EvalHandlerExecutionTime = Field(
        description="When evaluations should be performed in terms of where in the code / when during a pipeline run."
    )

    models: Literal["matrix", "most_recent"] = Field(
        "most_recent",
        description="For a evaluation interval given by the strategy, this selects what models to evaluate there.",
    )

    strategy: EvalStrategyConfig = Field(
        ..., description="Defining the strategy which obtains the intervals for the evaluation."
    )

    datasets: List[str] = Field(
        description=(
            "List of dataset references defined at top level of evaluation config, to enable running one "
            "handler on multiple datasets (e.g. train, test, combined)"
        )
    )
    """
    this makes sense as e.g. the BetweenTwoTriggersEvalStrategy can actually run on the training data.
    we could default this to use run the handler on every of the datasets in the root setting
    """
