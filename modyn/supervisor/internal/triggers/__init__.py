"""Supervisor module. The supervisor initiates a pipeline and coordinates all components.

"""

import os

from .datadrifttrigger import DataDriftTrigger  # noqa: F401
from .amounttrigger import DataAmountTrigger  # noqa: F401
from .timetrigger import TimeTrigger  # noqa: F401
from .trigger import Trigger  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
