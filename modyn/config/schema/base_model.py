from __future__ import annotations

from pydantic import BaseModel


class ModynBaseModel(BaseModel):
    class Config:
        extra = "forbid"
