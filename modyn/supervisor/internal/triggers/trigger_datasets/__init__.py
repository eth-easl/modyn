"""Supervisor module. The supervisor initiates a pipeline and coordinates all components.

"""

import os

from .dataloader_info import DataLoaderInfo  # noqa: F401
from .online_trigger_dataset import OnlineTriggerDataset  # noqa: F401
from .trigger_dataset_given_keys import TriggerDatasetGivenKeys  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
