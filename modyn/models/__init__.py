"""Models."""

import os

from .articlenet.articlenet import ArticleNet  # noqa: F401
from .dlrm.dlrm import DLRM  # noqa: F401
from .dummy.dummy import Dummy  # noqa: F401
from .fmownet.fmownet import FmowNet  # noqa: F401
from .gpt2.gpt2 import Gpt2  # noqa: F401
from .modular_adapters import modular_adapters  # noqa: F401
from .modular_adapters.modular_adapters import apply_kadapter, apply_lora  # noqa: F401
from .resnet18.resnet18 import ResNet18  # noqa: F401
from .resnet50.resnet50 import ResNet50  # noqa: F401
from .resnet152.resnet152 import ResNet152  # noqa: F401
from .rho_loss_twin_model.rho_loss_twin_model import RHOLOSSTwinModel  # noqa: F401
from .smallyearbooknet.smallyearbooknet import SmallYearbookNet  # noqa: F401
from .yearbooknet.yearbooknet import YearbookNet  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
