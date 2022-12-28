"""
This package contains the file system wrapper classes. The file system wrapper classes are used to abstract the file system
operations. This allows the storage module to be used with different file systems.
"""
import os
files = os.listdir(os.path.dirname(__file__))
files.remove('__init__.py')
__all__ = [f[:-3] for f in files if f.endswith(".py")]