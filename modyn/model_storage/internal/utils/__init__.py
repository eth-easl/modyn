"""Model Storage module.

The model storage module contains all classes and functions related to
the storage and retrieval of models.
"""

import os

from .model_storage_policy import ModelStoragePolicy  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
