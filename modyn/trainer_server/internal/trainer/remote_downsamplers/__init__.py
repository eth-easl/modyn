import os

from .remote_craig_downsampling import RemoteCraigDownsamplingStrategy  # noqa: F401
from .remote_grad_match_downsampling_strategy import RemoteGradMatchDownsamplingStrategy  # noqa: F401
from .remote_gradnorm_downsampling import RemoteGradNormDownsampling  # noqa: F401
from .remote_kcenter_greedy_downsampling_strategy import RemoteKcenterGreedyDownsamplingStrategy  # noqa: F401
from .remote_loss_downsampling import RemoteLossDownsampling  # noqa: F401
from .remote_rho_loss_downsampling import RemoteRHOLossDownsampling  # noqa: F401
from .remote_rs2_downsampling import RemoteRS2Downsampling  # noqa: F401
from .remote_submodular_downsampling_strategy import RemoteSubmodularDownsamplingStrategy  # noqa: F401
from .remote_uncertainty_downsampling_strategy import RemoteUncertaintyDownsamplingStrategy  # noqa: F401
from .remote_token_uncertainty_downsampling import RemoteTokenUncertaintyDownsampling  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
