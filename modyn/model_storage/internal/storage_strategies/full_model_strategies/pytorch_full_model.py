import logging
import pathlib

import torch
from modyn.model_storage.internal.storage_strategies.full_model_strategies.abstract_full_model_strategy import (
    AbstractFullModelStrategy,
)

logger = logging.getLogger(__name__)


class PyTorchFullModel(AbstractFullModelStrategy):
    """
    This full model strategy naively stores the whole model on disk (default pytorch implementation).
    """

    # pylint: disable-next=unused-argument
    def __init__(self, zipping_dir: pathlib.Path, zip_activated: bool, zip_algorithm_name: str, config: dict):
        super().__init__(zipping_dir, False, zip_algorithm_name)

        if zip_activated:
            logger.warning("The zipping option is disabled for this strategy since its already performed natively.")

    def _store_model(self, model_state: dict, file_path: pathlib.Path) -> None:
        torch.save(model_state, file_path)

    def _load_model(self, base_model_state: dict, file_path: pathlib.Path) -> dict:
        base_model_state.update(torch.load(file_path))
        return base_model_state
