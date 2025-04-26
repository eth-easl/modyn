import os

from .abstract_downsampling_strategy import AbstractDownsamplingStrategy  # noqa: F401
from .craig_downsampling_strategy import CraigDownsamplingStrategy  # noqa: F401
from .downsampling_scheduler import DownsamplingScheduler, instantiate_scheduler  # noqa: F401
from .gradmatch_downsampling_strategy import GradMatchDownsamplingStrategy  # noqa: F401
from .gradnorm_downsampling_strategy import GradNormDownsamplingStrategy  # noqa: F401
from .kcentergreedy_downsampling_strategy import KcenterGreedyDownsamplingStrategy  # noqa: F401
from .loss_downsampling_strategy import LossDownsamplingStrategy  # noqa: F401
from .no_downsampling_strategy import NoDownsamplingStrategy  # noqa: F401
from .rho_loss_downsampling_strategy import RHOLossDownsamplingStrategy  # noqa: F401
from .rs2_downsampling_strategy import RS2DownsamplingStrategy  # noqa: F401
from .submodular_downsampling_strategy import SubmodularDownsamplingStrategy  # noqa: F401
from .token_uncertainty_downsampling import TokenUncertaintyDownsamplingStrategy  # noqa: F401
from .uncertainty_downsampling_strategy import UncertaintyDownsamplingStrategy  # noqa: F401
from .utils import instantiate_downsampler  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
