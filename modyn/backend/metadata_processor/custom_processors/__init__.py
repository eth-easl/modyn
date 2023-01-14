"""
This module containes extensions of the MetadataProcessorStrategy class that
implement custom processing strategies.
"""
import os
files = os.listdir(os.path.dirname(__file__))
files.remove('__init__.py')
__all__ = [f[:-3] for f in files if f.endswith(".py")]
=======
"""
TODO: Describe what is in this directory/submodule.
"""

import os

# flake8: noqa
from .processor import base

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
>>>>>>> main:modyn/backend/ptmp/__init__.py
