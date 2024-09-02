import io
import json
import logging
import pathlib

import torch

from modyn.models.coreset_methods_support import CoresetSupportingModule
from modyn.utils import dynamic_module_import

logger = logging.getLogger(__name__)


class StatefulModel:
    """The StatefulModel stores a model and loads its state and metadata.

    DataDriftTrigger uses StatefulModel to create embeddings through a
    forward-pass on the wrapped model.
    """

    def __init__(
        self,
        model_id: int,
        model_class_name: str,
        model_config: str,
        device: str,
        amp: bool,
    ):
        self.model_id = model_id
        self.device = device
        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        self.amp = amp

        self.model_class_name = model_class_name
        model_module = dynamic_module_import("modyn.models")
        self.model_handler = getattr(model_module, model_class_name)
        assert self.model_handler is not None

        self.model_configuration_dict = json.loads(model_config)

        self.model = self.model_handler(self.model_configuration_dict, device, amp)
        assert self.model is not None
        # The model must be able to record embeddings.
        assert isinstance(self.model.model, CoresetSupportingModule)

    def _load_state(self, path: pathlib.Path) -> None:
        assert path.exists(), "Cannot load state from non-existing file"

        logger.info(f"Loading model state from {path}")
        with open(path, "rb") as state_file:
            checkpoint = torch.load(io.BytesIO(state_file.read()), map_location=torch.device("cpu"))

        assert "model" in checkpoint
        self.model.model.load_state_dict(checkpoint["model"])

        # delete trained model from disk
        path.unlink()
