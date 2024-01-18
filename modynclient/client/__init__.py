"""Client module. The client communicates with the modyn supervisor server.

"""
import os

from .client import Client  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
