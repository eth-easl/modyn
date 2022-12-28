"""
Data
"""
# flake8: noqa
import os
from .mnist_dataset import MNISTDataset

files = os.listdir(os.path.dirname(__file__))
files.remove('__init__.py')
__all__ = [f[:-3] for f in files if f.endswith(".py")]
