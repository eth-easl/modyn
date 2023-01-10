"""This module contains the file wrapper classes for the internal storage module.

The file wrapper classes are used to abstract the file operations.
This allows the storage module to be used with different file formats.
"""
import os
files = os.listdir(os.path.dirname(__file__))
files.remove('__init__.py')
__all__ = [f[:-3] for f in files if f.endswith(".py")]
