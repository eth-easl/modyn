import shutil
import tempfile
from pathlib import Path

import pytest
from modyn.backend.selector.internal.trigger_sample import (
    get_training_set_partition,
    get_trigger_samples,
    save_trigger_sample,
)

TMP_DIR = tempfile.mkdtemp()


@pytest.fixture(autouse=True)
def setup_and_teardown():
    Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

    yield

    shutil.rmtree(TMP_DIR)


def test_save_trigger_sample():
    save_trigger_sample(1, 2, 3, [(1, 1.0), (2, 2.0)], 4, TMP_DIR)

    with open(f"{TMP_DIR}/1_2_3_4.txt", "r", encoding="utf-8") as file:
        while line := file.readline().rstrip():
            assert line in ["1:1.0", "2:2.0"]


def test_get_trigger_samples():
    with open(f"{TMP_DIR}/1_2_3_4.txt", "w", encoding="utf-8") as file:
        file.write("1:1.0\n2:2.0\n3:3.0\n4:4.0")
    with open(f"{TMP_DIR}/1_2_3_5.txt", "w", encoding="utf-8") as file:
        file.write("3:3.0\n4:4.0\n5:5.0\n6:6.0")

    assert set(get_trigger_samples(1, 2, 3, TMP_DIR)) == {
        (1, 1.0),
        (2, 2.0),
        (3, 3.0),
        (4, 4.0),
        (3, 3.0),
        (4, 4.0),
        (5, 5.0),
        (6, 6.0),
    }

    expected_order = get_trigger_samples(1, 2, 3, TMP_DIR)

    result = get_trigger_samples(1, 2, 3, TMP_DIR, 0, 4, 8)
    assert len(result) == 2
    assert result == [expected_order[0], expected_order[1]]

    result = get_trigger_samples(1, 2, 3, TMP_DIR, 3, 4, 8)
    assert len(result) == 2
    assert result == [expected_order[6], expected_order[7]]

    result = get_trigger_samples(1, 2, 3, TMP_DIR, 1, 3, 8)
    assert len(result) == 3
    assert result == [expected_order[3], expected_order[4], expected_order[5]]


def test_get_trigger_samples_no_file():
    assert not get_trigger_samples(1, 2, 3, TMP_DIR)


def test_get_training_set_partition():
    training_samples = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert get_training_set_partition(0, 3, len(training_samples)) == (0, 4)

    assert get_training_set_partition(1, 3, len(training_samples)) == (4, 4)
    assert get_training_set_partition(2, 3, len(training_samples)) == (8, 2)

    with pytest.raises(ValueError):
        get_training_set_partition(3, 3, len(training_samples))
    with pytest.raises(ValueError):
        get_training_set_partition(-1, 3, len(training_samples))

    assert get_training_set_partition(0, 2, len(training_samples)) == (0, 5)
