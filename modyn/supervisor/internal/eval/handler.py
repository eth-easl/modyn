from __future__ import annotations

from dataclasses import dataclass

from modyn.config.schema.pipeline import EvalHandlerConfig


@dataclass
class EvalHandler:
    """Handles one series of evaluations configured through an `EvalHandlerConfig`.

    We will add more functionality to this class in the future.
    """

    config: EvalHandlerConfig
