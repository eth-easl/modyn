from __future__ import annotations

from dataclasses import dataclass

from modyn.config.schema.pipeline.evaluation import EvalHandlerConfig
from modyn.supervisor.internal.eval.triggers.eval_trigger import EvalTrigger


@dataclass
class EvalHandler:
    config: EvalHandlerConfig
    trigger: EvalTrigger | None
