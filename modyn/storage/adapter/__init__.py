"""
TODO: Describe what is in this directory/submodule.
"""
# flake8: noqa
from .dbm_adapter import DBMAdapter
from .s3_adapter import S3Adapter
from .postgresql_adapter import PostgreSQLAdapter
from .sqlite_adapter import SQLiteAdapter
from .base import BaseAdapter
import os

files = os.listdir(os.path.dirname(__file__))
files.remove('__init__.py')
__all__ = [f[:-3] for f in files if f.endswith(".py")]