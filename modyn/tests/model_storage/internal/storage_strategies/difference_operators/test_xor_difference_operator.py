import io

import torch
from modyn.model_storage.internal.storage_strategies import AbstractDifferenceOperator
from modyn.model_storage.internal.storage_strategies.difference_operators import XorDifferenceOperator


def test_inheritance():
    assert issubclass(XorDifferenceOperator.__class__, AbstractDifferenceOperator.__class__)


def test_calculate_difference():
    ones = torch.ones(1, dtype=torch.int32)

    difference_operator = XorDifferenceOperator()
    assert difference_operator.calculate_difference(ones, ones) == b"\x00\x00\x00\x00"

    twos = ones * 2
    assert difference_operator.calculate_difference(twos, ones) == b"\x03\x00\x00\x00"


def test_calculate_restore():
    difference_operator = XorDifferenceOperator()

    ones = torch.ones(1, dtype=torch.int32)
    buf = io.BytesIO()
    buf.write(b"\x00\x00\x00\x00")
    buf.seek(0)

    assert difference_operator.restore(ones, buf).item() == 1

    buf.seek(0)
    buf.write(b"\x03\x00\x00\x00")
    buf.seek(0)

    assert difference_operator.restore(ones, buf).item() == 2
