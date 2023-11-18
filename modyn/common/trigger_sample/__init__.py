"""
This submodule implements necessary functions to persist a calculated trigger training set to disk.
"""
import os

from .trigger_sample_storage import ArrayWrapper, TriggerSampleStorage  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
