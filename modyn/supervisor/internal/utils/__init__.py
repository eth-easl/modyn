"""Supervisor module. The supervisor initiates a pipeline and coordinates all components.

"""
import os

from .evaluation_status_tracker import EvaluationStatusTracker  # noqa: F401
from .supervisor_counter import SupervisorCounter  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
