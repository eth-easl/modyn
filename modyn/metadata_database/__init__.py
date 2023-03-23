"""This package contains the database classes for the metadata module.

The models are used to abstract the database operations.
This allows the storage module to be used with different databases.
"""
import os

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
