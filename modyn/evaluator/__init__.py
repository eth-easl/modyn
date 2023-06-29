"""
Evaluator module.

The evaluator module contains all classes and functions related the evaluation of models.
"""

import os

from .evaluator import Evaluator  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
