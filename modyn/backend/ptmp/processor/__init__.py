"""
TODO: Describe what is in this directory/submodule.
"""

# flake8: noqa
from .base import PostTrainingMetadataProcessor
from .simple_processor import SimpleProcessor
import os

files = os.listdir(os.path.dirname(__file__))
files.remove('__init__.py')
__all__ = [f[:-3] for f in files if f.endswith(".py")]
