import pathlib
import tempfile
from zipfile import ZIP_LZMA

import pytest
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


class MockComplexModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._bias = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))
        self._weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward(self, data):
        return data


def get_mock_model_after() -> MockModel:
    model_after = MockModel()
    model_after._weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

    return model_after


def test_init():
    incremental_strategy = WeightsDifference(
        zipping_dir=pathlib.Path(), zip_activated=False, zip_algorithm_name="", config={}
    )

    assert isinstance(incremental_strategy.difference_operator, SubDifferenceOperator.__class__)
    assert not incremental_strategy.split_exponent

    incremental_strategy = WeightsDifference(
        zipping_dir=pathlib.Path(),
        zip_activated=False,
        zip_algorithm_name="",
        config={"operator": "xor", "split_exponent": True},
    )

    assert not incremental_strategy.zip
    assert isinstance(incremental_strategy.difference_operator, XorDifferenceOperator.__class__)
    assert incremental_strategy.split_exponent

    incremental_strategy = WeightsDifference(
        zipping_dir=pathlib.Path(),
        zip_activated=True,
        zip_algorithm_name="ZIP_LZMA",
        config={"operator": "sub", "split_exponent": False},
    )

    assert incremental_strategy.zip
    assert incremental_strategy.zip_algorithm == ZIP_LZMA
    assert isinstance(incremental_strategy.difference_operator, SubDifferenceOperator.__class__)
    assert not incremental_strategy.split_exponent


def test_store_model():
    model_before = MockModel()
    model_after = get_mock_model_after()

    for operator in ["xor", "sub"]:
        incremental_strategy = WeightsDifference(
            zipping_dir=pathlib.Path(), zip_activated=False, zip_algorithm_name="", config={"operator": operator}
        )

        with tempfile.NamedTemporaryFile() as temporary_file:
            temp_file_path = pathlib.Path(temporary_file.name)

            incremental_strategy.store_model(model_after.state_dict(), model_before.state_dict(), temp_file_path)

            with open(temp_file_path, "rb") as stored_model_file:
                assert stored_model_file.read() == b"\x00\x00\x80\x3f\x00\x00\x80\x3f"


def test_load_model():
    with tempfile.NamedTemporaryFile() as temporary_file:
        temp_file_path = pathlib.Path(temporary_file.name)

        with open(temp_file_path, "wb") as stored_model_file:
            stored_model_file.write(b"\x00\x00\x80\x3f\x00\x00\x80\x3f")

        for operator in ["xor", "sub"]:
            incremental_strategy = WeightsDifference(
                zipping_dir=pathlib.Path(), zip_activated=False, zip_algorithm_name="", config={"operator": operator}
            )

            model = MockModel()
            model_state = incremental_strategy.load_model(model.state_dict(), temp_file_path)

            assert model_state["_weight"][0] == 1  # pylint: disable=unsubscriptable-object


def test_rle():
    assert WeightsDifference.encode_bytes(b"") == b""

    encoded = WeightsDifference.encode_bytes(b"\x00\x00\x02\x01\x01\x01\x00")
    assert encoded == b"\x02\x00\x01\x02\x03\x01\x01\x00"

    encoded = WeightsDifference.encode_bytes(512 * b"\x00" + b"\x01")
    assert encoded == b"\xff\x00\xff\x00\x02\x00\x01\x01"


def test_inv_rle():
    assert WeightsDifference.decode_bytes(b"") == b""

    encoded = WeightsDifference.decode_bytes(b"\x02\x00\x01\x02\x03\x01\x01\x00")
    assert encoded == b"\x00\x00\x02\x01\x01\x01\x00"

    encoded = WeightsDifference.decode_bytes(b"\xff\x00\xff\x00\x02\x00\x01\x01")
    assert encoded == 512 * b"\x00" + b"\x01"

    with pytest.raises(AssertionError):
        WeightsDifference.decode_bytes(b"\x02\x00\x01")


def test_store_then_load_model():
    model_before = MockComplexModel()
    before_state = model_before.state_dict()
    model_after = MockComplexModel()
    model_after._weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32) * 2)

    incremental_strategy = WeightsDifference(
        zipping_dir=pathlib.Path(),
        zip_activated=False,
        zip_algorithm_name="",
        config={"operator": "xor", "split_exponent": True, "rle": True},
    )

    with tempfile.NamedTemporaryFile() as temporary_file:
        temp_file_path = pathlib.Path(temporary_file.name)

        incremental_strategy.store_model(model_after.state_dict(), before_state, temp_file_path)

        with open(temp_file_path, "rb") as stored_model_file:
            # we store 2 exponent bytes.
            assert stored_model_file.read(8) == b"\x00\x00\x00\x00\x00\x00\x00\x02"

            # twice the xor difference between 2 and 1 in the exponent byte.
            assert stored_model_file.read(2) == b"\x02\xff"

            # xor difference of the float16 tensors.
            assert stored_model_file.read(4) == b"\x00\x00\x00\x00"

            # xor difference of the remaining float32 bytes.
            assert stored_model_file.read(8) == b"\x00\x00\x00\x00\x00\x00"

        state_dict = incremental_strategy.load_model(before_state, temp_file_path)

        assert state_dict["_bias"][0].item() == 1  # pylint: disable=unsubscriptable-object
        assert state_dict["_weight"][0].item() == 2  # pylint: disable=unsubscriptable-object
