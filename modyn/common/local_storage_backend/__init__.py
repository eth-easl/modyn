"""
This submodule implements necessary functions to persist samples to storage.
"""

import os

from .local_storage_backend import LocalStorageBackend  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
