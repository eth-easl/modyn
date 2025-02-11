"""Evaluator module.

The evaluator module contains all classes and functions related the
evaluation of models.
"""

import os

from .evaluation_info import EvaluationInfo  # noqa: F401
from .evaluation_process_info import EvaluationProcessInfo  # noqa: F401
from .evaluator_messages import EvaluatorMessages  # noqa: F401
from .tuning_info import TuningInfo  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
