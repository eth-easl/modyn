"""
Online dataset and utils
"""
# flake8: noqa
import os
from .online_dataset import OnlineDataset

files = os.listdir(os.path.dirname(__file__))
files.remove('__init__.py')
__all__ = [f[:-3] for f in files if f.endswith(".py")]
