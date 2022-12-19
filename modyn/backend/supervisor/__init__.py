"""Supervisor module. The supervisor initiates a pipeline and coordinates all components.

"""
from .supervisor import Supervisor  # noqa: F401
import os
files = os.listdir(os.path.dirname(__file__))
files.remove('__init__.py')
__all__ = [f[:-3] for f in files if f.endswith(".py")]
