"""Models.

"""
import os

from .resnet18.resnet18 import ResNet18  # noqa: F401
from .dlrm.dlrm import DLRM # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
