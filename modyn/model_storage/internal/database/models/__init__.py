"""Contains the ORM models for the model storage component
"""
import os

from .trained_model import TrainedModel  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
