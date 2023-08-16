import pathlib

from modyn.model_storage.internal.storage_strategies.full_model_strategies import AbstractFullModelStrategy
from modyn.model_storage.internal.utils import read_tensor_from_bytes


class CompressedFullModel(AbstractFullModelStrategy):
    """
    This full model strategy stores the weights as binary sequence.
    """

    def _save_model(self, model_state: dict, file_path: pathlib.Path) -> None:
        with open(file_path, "wb") as file:
            for _, tensor in model_state.items():
                file.write(tensor.numpy().tobytes())

    def _load_model(self, base_model_state: dict, file_path: pathlib.Path) -> None:
        with open(file_path, "rb") as file:
            for layer, tensor in base_model_state.items():
                base_model_state[layer] = read_tensor_from_bytes(tensor, file)

    def validate_config(self, config: dict) -> None:
        pass
