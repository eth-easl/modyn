import shutil
import tempfile
from pathlib import Path

import pytest
from modyn.backend.selector.internal.trigger_sample import TriggerSampleStorage

TMP_DIR = tempfile.mkdtemp()


@pytest.fixture(autouse=True)
def setup_and_teardown():
    Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

    yield

    shutil.rmtree(TMP_DIR)


def test_save_trigger_sample():
    TriggerSampleStorage(TMP_DIR).save_trigger_sample(1, 2, 3, [(1, 1.0), (2, 2.0)], 4)

    samples = TriggerSampleStorage(TMP_DIR).get_trigger_samples(1, 2, 3)

    assert len(samples) == 2
    assert samples[0] == (1, 1.0)
    assert samples[1] == (2, 2.0)


def test_get_file_size():
    TriggerSampleStorage(TMP_DIR).save_trigger_sample(1, 2, 3, [(1, 1.0), (2, 2.0)], 4)

    file_path = Path(TMP_DIR) / "1_2_3_4"
    TriggerSampleStorage(TMP_DIR)._get_file_size(file_path)


def test_parse_file_subset():
    TriggerSampleStorage(TMP_DIR).save_trigger_sample(1, 2, 3, [(1, 1.0), (2, 2.0)], 4)

    file_path = Path(TMP_DIR) / "1_2_3_4"
    samples = TriggerSampleStorage(TMP_DIR)._parse_file_subset(file_path, 0, 1)

    assert len(samples) == 1
    assert samples[0] == (1, 1.0)

    samples = TriggerSampleStorage(TMP_DIR)._parse_file_subset(file_path, 0, 2)
    assert len(samples) == 2
    assert samples[0] == (1, 1.0)
    assert samples[1] == (2, 2.0)

    TriggerSampleStorage(TMP_DIR).save_trigger_sample(1, 2, 3, [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)], 5)
    file_path = Path(TMP_DIR) / "1_2_3_5"

    samples = TriggerSampleStorage(TMP_DIR)._parse_file_subset(file_path, 0, 3)
    assert len(samples) == 3
    assert samples[0] == (1, 1.0)
    assert samples[1] == (2, 2.0)
    assert samples[2] == (3, 3.0)

    samples = TriggerSampleStorage(TMP_DIR)._parse_file_subset(file_path, 1, 3)
    assert len(samples) == 2
    assert samples[0] == (2, 2.0)
    assert samples[1] == (3, 3.0)

    TriggerSampleStorage(TMP_DIR).save_trigger_sample(1, 2, 3, [(1, 1.0), (2, 2.0)], 5)
    with pytest.raises(ValueError):
        _ = TriggerSampleStorage(TMP_DIR)._parse_file_subset(file_path, 0, 3)


def test_parse_file():
    TriggerSampleStorage(TMP_DIR).save_trigger_sample(1, 2, 3, [(1, 1.0), (2, 2.0)], 4)
    file_path = Path(TMP_DIR) / "1_2_3_4"

    samples = TriggerSampleStorage(TMP_DIR)._parse_file(file_path)
    assert len(samples) == 2
    assert samples[0] == (1, 1.0)
    assert samples[1] == (2, 2.0)

    TriggerSampleStorage(TMP_DIR).save_trigger_sample(1, 2, 3, [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)], 5)
    file_path = Path(TMP_DIR) / "1_2_3_5"

    samples = TriggerSampleStorage(TMP_DIR)._parse_file(file_path)
    assert len(samples) == 4
    assert samples[0] == (1, 1.0)
    assert samples[1] == (2, 2.0)
    assert samples[2] == (3, 3.0)
    assert samples[3] == (4, 4.0)


def test_get_trigger_samples():
    TriggerSampleStorage(TMP_DIR).save_trigger_sample(1, 2, 3, [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)], 4)
    TriggerSampleStorage(TMP_DIR).save_trigger_sample(1, 2, 3, [(3, 3.0), (4, 4.0), (5, 5.0), (6, 6.0)], 5)

    expected_order = [
        (1, 1.0),
        (2, 2.0),
        (3, 3.0),
        (4, 4.0),
        (3, 3.0),
        (4, 4.0),
        (5, 5.0),
        (6, 6.0),
    ]

    assert TriggerSampleStorage(TMP_DIR).get_trigger_samples(1, 2, 3) == expected_order

    result = TriggerSampleStorage(TMP_DIR).get_trigger_samples(1, 2, 3, 0, 4, 8)
    assert len(result) == 2
    assert result == [expected_order[0], expected_order[1]]

    result = TriggerSampleStorage(TMP_DIR).get_trigger_samples(1, 2, 3, 1, 4, 8)
    assert len(result) == 2
    assert result == [expected_order[2], expected_order[3]]

    result = TriggerSampleStorage(TMP_DIR).get_trigger_samples(1, 2, 3, 2, 4, 8)
    assert len(result) == 2
    assert result == [expected_order[4], expected_order[5]]

    result = TriggerSampleStorage(TMP_DIR).get_trigger_samples(1, 2, 3, 3, 4, 8)
    assert len(result) == 2
    assert result == [expected_order[6], expected_order[7]]

    result = TriggerSampleStorage(TMP_DIR).get_trigger_samples(1, 2, 3, 1, 3, 8)
    assert len(result) == 3
    assert result == [expected_order[3], expected_order[4], expected_order[5]]

    result = TriggerSampleStorage(TMP_DIR).get_trigger_samples(1, 2, 3, 2, 3, 8)
    assert len(result) == 2
    assert result == [expected_order[6], expected_order[7]]

    result = TriggerSampleStorage(TMP_DIR).get_trigger_samples(1, 2, 3, 0, 1, 8)
    assert len(result) == 8
    assert result == expected_order

    result = TriggerSampleStorage(TMP_DIR).get_trigger_samples(1, 2, 3, 9, 10, 8)
    assert len(result) == 0
    assert result == []


def test_get_trigger_samples_no_file():
    assert not TriggerSampleStorage(TMP_DIR).get_trigger_samples(1, 2, 3)


def test_get_training_set_partition():
    training_samples = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert TriggerSampleStorage(TMP_DIR).get_training_set_partition(0, 3, len(training_samples)) == (0, 4)

    assert TriggerSampleStorage(TMP_DIR).get_training_set_partition(1, 3, len(training_samples)) == (4, 4)
    assert TriggerSampleStorage(TMP_DIR).get_training_set_partition(2, 3, len(training_samples)) == (8, 2)

    with pytest.raises(ValueError):
        TriggerSampleStorage(TMP_DIR).get_training_set_partition(3, 3, len(training_samples))
    with pytest.raises(ValueError):
        TriggerSampleStorage(TMP_DIR).get_training_set_partition(-1, 3, len(training_samples))

    assert TriggerSampleStorage(TMP_DIR).get_training_set_partition(0, 2, len(training_samples)) == (0, 5)

    training_samples = [1, 2, 3]
    assert TriggerSampleStorage(TMP_DIR).get_training_set_partition(0, 8, len(training_samples)) == (0, 1)
    assert TriggerSampleStorage(TMP_DIR).get_training_set_partition(1, 8, len(training_samples)) == (1, 1)
    assert TriggerSampleStorage(TMP_DIR).get_training_set_partition(2, 8, len(training_samples)) == (2, 1)
    assert TriggerSampleStorage(TMP_DIR).get_training_set_partition(3, 8, len(training_samples)) == (0, 0)
    assert TriggerSampleStorage(TMP_DIR).get_training_set_partition(4, 8, len(training_samples)) == (0, 0)
    assert TriggerSampleStorage(TMP_DIR).get_training_set_partition(5, 8, len(training_samples)) == (0, 0)
    assert TriggerSampleStorage(TMP_DIR).get_training_set_partition(6, 8, len(training_samples)) == (0, 0)
    assert TriggerSampleStorage(TMP_DIR).get_training_set_partition(7, 8, len(training_samples)) == (0, 0)
