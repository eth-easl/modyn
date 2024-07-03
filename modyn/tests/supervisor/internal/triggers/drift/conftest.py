from typing import Iterator

import numpy as np
import pytest


# back up seed before all tests
@pytest.fixture(autouse=True, scope="package")
def set_seed() -> Iterator[None]:
    seed_backup = np.random.get_state()
    np.random.seed(42)

    yield

    np.random.set_state(seed_backup)


@pytest.fixture(scope="package")
def data_ref() -> np.ndarray:
    return np.stack(
        [
            (x, y)
            for x, y in zip(
                np.random.normal(loc=0.0, scale=1.0, size=5000), np.random.normal(loc=0.0, scale=1.0, size=5000)
            )
        ]
    )


@pytest.fixture(scope="package")
def data_h0() -> np.ndarray:
    return np.stack(
        [
            (x, y)
            for x, y in zip(
                np.random.normal(loc=0.0, scale=1.0, size=100), np.random.normal(loc=0.0, scale=1.0, size=100)
            )
        ]
    )


@pytest.fixture(scope="module")
def data_cur() -> np.ndarray:
    return np.stack(
        [
            (x, y)
            for x, y in zip(
                np.random.normal(loc=0.2, scale=1.0, size=120), np.random.normal(loc=-0.3, scale=1.0, size=120)
            )
        ]
    )
