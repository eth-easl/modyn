"""
Model Storage module.

The model storage module contains all classes and functions related to the storage and retrieval of models.
"""

import os

from .abstract_full_model_strategy import AbstractFullModelStrategy  # noqa: F401
from .binary_full_model import BinaryFullModel  # noqa: F401
from .pytorch_full_model import PyTorchFullModel  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
