"""Storage module.

The storage module contains all classes and functions related to the storage and retrieval of data.
"""

import os

from .storage import Storage  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
