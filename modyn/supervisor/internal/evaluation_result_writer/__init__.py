"""Supervisor module. The supervisor initiates a pipeline and coordinates all components.

"""

import os

from .abstract_evaluation_result_writer import AbstractEvaluationResultWriter  # noqa: F401
from .json_result_writer import JsonResultWriter  # noqa: F401
from .tensorboard_result_writer import TensorboardResultWriter  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
