import os

from .abstract_downsampling_strategy import AbstractDownsamplingStrategy  # noqa: F401
from .empty_downsampling_strategy import EmptyDownsamplingStrategy  # noqa: F401
from .gradnorm_downsampling_strategy import GradNormDownsamplingStrategy  # noqa: F401
from .loss_downsampling_strategy import LossDownsamplingStrategy  # noqa: F401
from .scheduler_downsampling_strategy import SchedulerDownsamplingStrategy  # noqa: F401
from .utils import instantiate_downsampler  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
