"""Metrics module.

The metrics module contains all classes that can be used to evaluate a
model on a specific metric.
"""

import os

from .abstract_decomposable_metric import AbstractDecomposableMetric  # noqa: F401
from .abstract_evaluation_metric import AbstractEvaluationMetric  # noqa: F401
from .abstract_holistic_metric import AbstractHolisticMetric  # noqa: F401
from .accuracy import Accuracy  # noqa: F401
from .f1_score import F1Score  # noqa: F401
from .meteor import Meteor  # noqa: F401
from .perplexity import Perplexity  # noqa: F401
from .roc_auc import RocAuc  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
