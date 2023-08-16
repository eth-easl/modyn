import json
from zipfile import ZIP_DEFLATED, ZIP_LZMA

import pytest
from modyn.model_storage.internal.storage_strategies.difference_operators import XorDifferenceOperator
from modyn.model_storage.internal.utils import ModelStorageStrategy


def test_basic_model_storage_strategy():
    model_storage_strategy = ModelStorageStrategy("PyTorchFullModel", None, None, None)

    assert model_storage_strategy.incremental_model_strategy is None
    assert model_storage_strategy.full_model_interval is None
    assert not model_storage_strategy.full_model_strategy.zip


def test_extended_model_storage_strategy():
    model_storage_strategy = ModelStorageStrategy(
        full_model_strategy_name="PyTorchFullModel",
        full_model_strategy_zip=True,
        full_model_strategy_zip_algorithm="ZIP_LZMA",
        full_model_strategy_config=None,
    )
    model_storage_strategy.register_incremental_model_strategy(
        name="WeightsDifference",
        zip_enabled=True,
        zip_algorithm=None,
        config=json.dumps({"operator": "xor", "split_exponent": True}),
        full_model_interval=10,
    )

    assert model_storage_strategy.full_model_strategy.zip
    assert model_storage_strategy.full_model_strategy.zip_algorithm == ZIP_LZMA

    weights_diff_strategy = model_storage_strategy.incremental_model_strategy
    assert weights_diff_strategy.zip
    assert weights_diff_strategy.zip_algorithm == ZIP_DEFLATED
    assert getattr(weights_diff_strategy, "split_exponent")
    assert isinstance(getattr(weights_diff_strategy, "difference_operator"), XorDifferenceOperator.__class__)

    assert model_storage_strategy.full_model_interval == 10


def test_model_storage_strategy_invalid():
    strategy = ModelStorageStrategy(
        full_model_strategy_name="PyTorchFullModel",
        full_model_strategy_zip=None,
        full_model_strategy_zip_algorithm=None,
        full_model_strategy_config=None,
    )

    with pytest.raises(ValueError):
        strategy.register_incremental_model_strategy("WeightsDifference", None, None, None, 0)

    with pytest.raises(NotImplementedError):
        strategy.register_incremental_model_strategy("UnknownStrategy", None, None, None, None)
