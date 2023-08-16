import pathlib
import tempfile
from zipfile import ZIP_LZMA

import torch
from modyn.model_storage.internal.storage_strategies.difference_operators import (
    SubDifferenceOperator,
    XorDifferenceOperator,
)
from modyn.model_storage.internal.storage_strategies.incremental_model_strategies import WeightsDifference


class MockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(self, data):
        return data


def get_mock_model_after() -> MockModel:
    model_after = MockModel()
    model_after._weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

    return model_after


def test_init():
    incremental_strategy = WeightsDifference(zip_activated=False, zip_algorithm_name="", config={})

    assert isinstance(incremental_strategy.difference_operator, SubDifferenceOperator.__class__)
    assert not incremental_strategy.split_exponent

    incremental_strategy = WeightsDifference(
        zip_activated=False, zip_algorithm_name="", config={"operator": "xor", "split_exponent": True}
    )

    assert not incremental_strategy.zip
    assert isinstance(incremental_strategy.difference_operator, XorDifferenceOperator.__class__)
    assert incremental_strategy.split_exponent

    incremental_strategy = WeightsDifference(
        zip_activated=True, zip_algorithm_name="ZIP_LZMA", config={"operator": "sub", "split_exponent": False}
    )

    assert incremental_strategy.zip
    assert incremental_strategy.zip_algorithm == ZIP_LZMA
    assert isinstance(incremental_strategy.difference_operator, SubDifferenceOperator.__class__)
    assert not incremental_strategy.split_exponent


def test_save_model():
    model_before = MockModel()
    model_after = get_mock_model_after()

    for operator in ["xor", "sub"]:
        incremental_strategy = WeightsDifference(
            zip_activated=False, zip_algorithm_name="", config={"operator": operator}
        )

        with tempfile.NamedTemporaryFile() as temporary_file:
            temp_file_path = pathlib.Path(temporary_file.name)

            incremental_strategy.save_model(model_after.state_dict(), model_before.state_dict(), temp_file_path)

            with open(temp_file_path, "rb") as stored_model_file:
                assert stored_model_file.read() == b"\x00\x00\x80\x3f\x00\x00\x80\x3f"


def test_load_model():
    with tempfile.NamedTemporaryFile() as temporary_file:
        temp_file_path = pathlib.Path(temporary_file.name)

        with open(temp_file_path, "wb") as stored_model_file:
            stored_model_file.write(b"\x00\x00\x80\x3f\x00\x00\x80\x3f")

        for operator in ["xor", "sub"]:
            model = MockModel()
            model_state = model.state_dict()
            incremental_strategy = WeightsDifference(
                zip_activated=False, zip_algorithm_name="", config={"operator": operator}
            )

            incremental_strategy.load_model(model_state, temp_file_path)

            assert model_state["_weight"][0] == 1  # pylint: disable=unsubscriptable-object
