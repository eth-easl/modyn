"""Models.

"""
from .resnet18.resnet18 import ResNet18  # noqa: F401
import os
files = os.listdir(os.path.dirname(__file__))
files.remove('__init__.py')
__all__ = [f[:-3] for f in files if f.endswith(".py")]
