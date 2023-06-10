"""
This submodule contains extensions of the Selector class that implement
concrete selection strategies.
"""
import os

from .abstract_selection_strategy import AbstractSelectionStrategy  # noqa: F401
from .freshness_sampling_strategy import FreshnessSamplingStrategy  # noqa: F401
from .gradnorm_downsampling_strategy import GradNormDownsamplingStrategy  # noqa: F401
from .loss_downsampling_strategy import LossDownsamplingStrategy  # noqa: F401
from .new_data_strategy import NewDataStrategy  # noqa: F401
from .random_presampling_strategy import RandomPresamplingStrategy  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
