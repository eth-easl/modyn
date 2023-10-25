"""
Model Storage module.

The model storage module contains all classes and functions related to the storage and retrieval of models.
"""

import os

from .sub_difference_operator import SubDifferenceOperator  # noqa: F401
from .xor_difference_operator import XorDifferenceOperator  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
