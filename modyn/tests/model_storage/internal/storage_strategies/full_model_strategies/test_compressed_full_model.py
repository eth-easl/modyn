import pathlib
import tempfile

import torch
from modyn.model_storage.internal.storage_strategies.full_model_strategies import CompressedFullModel


class MockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward(self, data):
        return data


def test_store_model():
    model = MockModel()
    full_model_strategy = CompressedFullModel(
        zipping_dir=pathlib.Path(), zip_activated=False, zip_algorithm_name="", config={}
    )
    with tempfile.NamedTemporaryFile() as temporary_file:
        temp_file_path = pathlib.Path(temporary_file.name)

        full_model_strategy.store_model(model.state_dict(), temp_file_path)

        with open(temp_file_path, "rb") as stored_model_file:
            assert stored_model_file.read() == b"\x00\x00\x80\x3f\x00\x00\x80\x3f"


def test_load_model():
    model = MockModel()
    full_model_strategy = CompressedFullModel(
        zipping_dir=pathlib.Path(), zip_activated=False, zip_algorithm_name="", config={}
    )
    with tempfile.NamedTemporaryFile() as temporary_file:
        temp_file_path = pathlib.Path(temporary_file.name)

        with open(temp_file_path, "wb") as stored_model_file:
            assert stored_model_file.write(b"\x00\x00\x00\x3f\x00\x00\x00\x3f")

        state_dict = model.state_dict()
        full_model_strategy.load_model(state_dict, temp_file_path)

        assert state_dict["_weight"][0] == 0.5  # pylint: disable=unsubscriptable-object
