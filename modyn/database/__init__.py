"""
Abstract database connection class.
"""

import os

from .partition_by_meta import PartitionByMeta  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
