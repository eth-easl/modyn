import json
import pathlib
from zipfile import ZIP_LZMA

import pytest
from modyn.model_storage.internal.storage_strategies.difference_operators import XorDifferenceOperator
from modyn.model_storage.internal.utils import ModelStoragePolicy


def test_basic_model_storage_policy():
    policy = ModelStoragePolicy(pathlib.Path(), "PyTorchFullModel", None, None, None)

    assert policy.incremental_model_strategy is None
    assert policy.full_model_interval is None
    assert not policy.full_model_strategy.zip


def test_extended_model_storage_policy():
    policy = ModelStoragePolicy(
        zipping_dir=pathlib.Path(),
        full_model_strategy_name="PyTorchFullModel",
        full_model_strategy_zip=True,
        full_model_strategy_zip_algorithm=None,
        full_model_strategy_config=None,
    )
    policy.register_incremental_model_strategy(
        name="WeightsDifference",
        zip_enabled=True,
        zip_algorithm="ZIP_LZMA",
        config=json.dumps({"operator": "xor", "split_exponent": True}),
        full_model_interval=10,
    )

    assert policy.zipping_dir == pathlib.Path("")
    assert not policy.full_model_strategy.zip

    weights_diff_strategy = policy.incremental_model_strategy
    assert weights_diff_strategy.zip
    assert weights_diff_strategy.zip_algorithm == ZIP_LZMA
    assert getattr(weights_diff_strategy, "split_exponent")
    assert isinstance(getattr(weights_diff_strategy, "difference_operator"), XorDifferenceOperator.__class__)

    assert policy.full_model_interval == 10


def test_model_storage_policy_invalid():
    policy = ModelStoragePolicy(
        zipping_dir=pathlib.Path(),
        full_model_strategy_name="PyTorchFullModel",
        full_model_strategy_zip=None,
        full_model_strategy_zip_algorithm=None,
        full_model_strategy_config=None,
    )

    with pytest.raises(ValueError):
        policy.register_incremental_model_strategy("WeightsDifference", None, None, None, 0)

    with pytest.raises(NotImplementedError):
        policy.register_incremental_model_strategy("UnknownStrategy", None, None, None, None)
