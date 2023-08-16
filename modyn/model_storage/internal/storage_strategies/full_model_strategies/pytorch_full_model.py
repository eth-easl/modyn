import pathlib

import torch
from modyn.model_storage.internal.storage_strategies.full_model_strategies.abstract_full_model_strategy import (
    AbstractFullModelStrategy,
)


class PyTorchFullModel(AbstractFullModelStrategy):
    """
    This full model strategy naively stores the whole model on disk (default pytorch implementation).
    """

    def _save_model(self, model_state: dict, file_path: pathlib.Path) -> None:
        torch.save(model_state, file_path)

    def _load_model(self, base_model_state: dict, file_path: pathlib.Path) -> None:
        base_model_state.update(torch.load(file_path))

    def validate_config(self, config: dict) -> None:
        pass
