import os

from .remote_gradnorm_downsample import RemoteGradNormDownsampling  # noqa: F401
from .remote_loss_downsample import RemoteLossDownsampling  # noqa: F401


files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
