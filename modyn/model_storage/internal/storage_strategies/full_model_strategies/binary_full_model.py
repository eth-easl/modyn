import pathlib

from modyn.model_storage.internal.storage_strategies.full_model_strategies import AbstractFullModelStrategy
from modyn.utils import get_tensor_byte_size, reconstruct_tensor_from_bytes


class BinaryFullModel(AbstractFullModelStrategy):
    """
    This full model strategy stores the weights as binary sequence.
    """

    # pylint: disable-next=unused-argument
    def __init__(self, zipping_dir: pathlib.Path, zip_activated: bool, zip_algorithm_name: str, config: dict):
        super().__init__(zipping_dir, zip_activated, zip_algorithm_name)

    def _store_model(self, model_state: dict, file_path: pathlib.Path) -> None:
        with open(file_path, "wb") as file:
            for tensor in model_state.values():
                file.write(tensor.numpy().tobytes())

    def _load_model(self, base_model_state: dict, file_path: pathlib.Path) -> dict:
        with open(file_path, "rb") as file:
            for layer, tensor in base_model_state.items():
                num_bytes = get_tensor_byte_size(tensor)
                base_model_state[layer] = reconstruct_tensor_from_bytes(tensor, file.read(num_bytes))
        return base_model_state
