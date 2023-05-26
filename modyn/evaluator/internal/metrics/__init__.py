"""
Metrics module.

The metrics module contains all classes that can be used to evaluate a model on a specific metric.
"""

import os

from .abstract_evaluation_metric import AbstractEvaluationMetric  # noqa: F401
from .accuracy_metric import AccuracyMetric  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
