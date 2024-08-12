from __future__ import annotations

from pydantic import Field

from modyn.config.schema.base_model import ModynBaseModel


class _BaseWindowingStrategy(ModynBaseModel):
    allow_overlap: bool = Field(
        False,
        description=(
            "Whether the windows are allowed to overlap. This is useful for time-based windows."
            "If set to False, the current window will be reset after each trigger."
        ),
    )
