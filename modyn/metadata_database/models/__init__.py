"""This package contains all the ORM models for the database.

The models are used to abstract the database operations.
This allows the storage module to be used with different databases.
"""
import os

from .pipelines import Pipeline  # noqa: F401
from .sample_training_metadata import SampleTrainingMetadata  # noqa: F401
from .selector_state_metadata import SelectorStateMetadata  # noqa: F401
from .trigger_partitions import TriggerPartition  # noqa: F401
from .trigger_training_metadata import TriggerTrainingMetadata  # noqa: F401
from .triggers import Trigger  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
