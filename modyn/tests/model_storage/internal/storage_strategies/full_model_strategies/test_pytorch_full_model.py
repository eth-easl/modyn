import pathlib
import tempfile

import torch
from modyn.model_storage.internal.storage_strategies.full_model_strategies import PyTorchFullModel


class MockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward(self, data):
        return data


def test_store_model():
    full_model_strategy = PyTorchFullModel(
        zipping_dir=pathlib.Path(), zip_activated=False, zip_algorithm_name="", config={}
    )
    with tempfile.NamedTemporaryFile() as temporary_file:
        temp_file_path = pathlib.Path(temporary_file.name)

        model = MockModel()
        full_model_strategy.store_model(model.state_dict(), temp_file_path)

        loaded_state = torch.load(temp_file_path)

        assert loaded_state["_weight"][0] == 1.0


def test_load_model():
    full_model_strategy = PyTorchFullModel(
        zipping_dir=pathlib.Path(), zip_activated=False, zip_algorithm_name="", config={}
    )
    with tempfile.NamedTemporaryFile() as temporary_file:
        temp_file_path = pathlib.Path(temporary_file.name)

        model = MockModel()
        torch.save(model.state_dict(), temp_file_path)

        model._weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32) * 2)
        state_dict = full_model_strategy.load_model(model.state_dict(), temp_file_path)

        assert state_dict["_weight"][0] == 1.0  # pylint: disable=unsubscriptable-object


def test_store_then_load():
    with tempfile.NamedTemporaryFile() as temporary_file:
        temp_file_path = pathlib.Path(temporary_file.name)
        full_model_strategy = PyTorchFullModel(
            zipping_dir=temp_file_path.parent, zip_activated=False, zip_algorithm_name="", config={}
        )

        model = MockModel()
        full_model_strategy.store_model(model.state_dict(), temp_file_path)

        model._weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32) * 2)
        state_dict = full_model_strategy.load_model(model.state_dict(), temp_file_path)

        assert state_dict["_weight"][0] == 1.0  # pylint: disable=unsubscriptable-object
