"""Supervisor module.

The supervisor initiates a pipeline and coordinates all components.
"""

import os

from .embedding_encoder import EmbeddingEncoder  # noqa: F401
from .embedding_encoder_downloader import EmbeddingEncoderDownloader  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
