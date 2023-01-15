"""
This submodule provides the functionality for the Metadata Processor server component.
"""
import os

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
