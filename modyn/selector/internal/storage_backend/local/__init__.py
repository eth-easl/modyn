"""
This submodule provides backends for storing the seen samples during a pipeline.
"""
import os

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
