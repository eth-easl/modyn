import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from modyn.common.trigger_storage_cpp.trigger_storage_cpp import TriggerStorageCPP

TMP_DIR = tempfile.mkdtemp()


@pytest.fixture(autouse=True)
def setup_and_teardown():
    Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

    yield

    shutil.rmtree(TMP_DIR)


def test_save_trigger_sample():
    TriggerStorageCPP(TMP_DIR).save_trigger_sample(1, 2, 3, np.array([(1, 1.0), (2, 2.0)], dtype=np.dtype("i8,f8")), 4)

    samples = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3)

    assert len(samples) == 2
    assert samples[0] == (1, 1.0)
    assert samples[1] == (2, 2.0)


def test_get_file_size():
    TriggerStorageCPP(TMP_DIR).save_trigger_sample(1, 2, 3, np.array([(1, 1.0), (2, 2.0)], dtype=np.dtype("i8,f8")), 4)

    file_path = Path(TMP_DIR) / "1_2_3_4.npy"
    TriggerStorageCPP(TMP_DIR)._get_num_samples_in_file(file_path)


def test_parse_file_subset():
    TriggerStorageCPP(TMP_DIR).save_trigger_sample(1, 2, 3, np.array([(1, 1.0), (2, 2.0)], dtype=np.dtype("i8,f8")), 4)

    file_path = Path(TMP_DIR) / "1_2_3_4.npy"
    samples = TriggerStorageCPP(TMP_DIR)._parse_file_subset(file_path, 0, 1)

    assert len(samples) == 1
    assert samples[0][0] == 1
    assert samples[0][1] == 1.0

    samples = TriggerStorageCPP(TMP_DIR)._parse_file_subset(file_path, 0, 2)
    assert len(samples) == 2
    assert samples[0][0] == 1
    assert samples[0][1] == 1.0
    assert samples[1][0] == 2
    assert samples[1][1] == 2.0

    TriggerStorageCPP(TMP_DIR).save_trigger_sample(
        1,
        2,
        3,
        np.array([(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)], dtype=np.dtype("i8,f8")),
        5,
    )
    file_path = Path(TMP_DIR) / "1_2_3_5.npy"

    samples = TriggerStorageCPP(TMP_DIR)._parse_file_subset(file_path, 0, 3)
    assert len(samples) == 3
    assert samples[0][0] == 1
    assert samples[0][1] == 1.0
    assert samples[1][0] == 2
    assert samples[1][1] == 2.0
    assert samples[2][0] == 3
    assert samples[2][1] == 3.0

    samples = TriggerStorageCPP(TMP_DIR)._parse_file_subset(file_path, 1, 3)
    assert len(samples) == 2
    assert samples[0][0] == 2
    assert samples[0][1] == 2.0
    assert samples[1][0] == 3
    assert samples[1][1] == 3.0

    TriggerStorageCPP(TMP_DIR).save_trigger_sample(1, 2, 3, np.array([(1, 1.0), (2, 2.0)], dtype=np.dtype("i8,f8")), 5)
    with pytest.raises(IndexError):
        _ = TriggerStorageCPP(TMP_DIR)._parse_file_subset(file_path, 0, 3)


def test_parse_file():
    TriggerStorageCPP(TMP_DIR).save_trigger_sample(1, 2, 3, np.array([(1, 1.0), (2, 2.0)], dtype=np.dtype("i8,f8")), 4)
    file_path = Path(TMP_DIR) / "1_2_3_4.npy"

    samples = TriggerStorageCPP(TMP_DIR)._parse_file(file_path)
    assert len(samples) == 2
    assert samples[0][0] == 1
    assert samples[0][1] == 1.0
    assert samples[1][0] == 2
    assert samples[1][1] == 2.0

    TriggerStorageCPP(TMP_DIR).save_trigger_sample(
        1,
        2,
        3,
        np.array([(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)], dtype=np.dtype("i8,f8")),
        5,
    )
    file_path = Path(TMP_DIR) / "1_2_3_5.npy"

    samples = TriggerStorageCPP(TMP_DIR)._parse_file(file_path)
    assert len(samples) == 4
    assert samples[0][0] == 1
    assert samples[0][1] == 1.0
    assert samples[1][0] == 2
    assert samples[1][1] == 2.0
    assert samples[2][0] == 3
    assert samples[2][1] == 3.0
    assert samples[3][0] == 4
    assert samples[3][1] == 4.0


def test_get_trigger_samples():
    TriggerStorageCPP(TMP_DIR).save_trigger_sample(
        1,
        2,
        3,
        np.array([(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)], dtype=np.dtype("i8,f8")),
        4,
    )
    TriggerStorageCPP(TMP_DIR).save_trigger_sample(
        1,
        2,
        3,
        np.array([(3, 3.0), (4, 4.0), (5, 5.0), (6, 6.0)], dtype=np.dtype("i8,f8")),
        5,
    )
    TriggerStorageCPP(TMP_DIR).save_trigger_sample(
        1,
        2,
        3,
        np.array([(7, 7.0), (8, 8.0), (8, 8.0), (8, 8.0)], dtype=np.dtype("i8,f8")),
        6,
    )
    TriggerStorageCPP(TMP_DIR).save_trigger_sample(
        1,
        2,
        3,
        np.array([(9, 9.0), (10, 10.0), (11, 11.0), (12, 12.0)], dtype=np.dtype("i8,f8")),
        7,
    )

    expected_order = [
        (1, 1.0),
        (2, 2.0),
        (3, 3.0),
        (4, 4.0),
        (3, 3.0),
        (4, 4.0),
        (5, 5.0),
        (6, 6.0),
        (7, 7.0),
        (8, 8.0),
        (8, 8.0),
        (8, 8.0),
        (9, 9.0),
        (10, 10.0),
        (11, 11.0),
        (12, 12.0),
    ]

    assert TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3) == expected_order

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 0, 4, 16)
    assert len(result) == 4
    assert result == expected_order[0:4]

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 1, 4, 16)
    assert len(result) == 4
    assert result == expected_order[4:8]

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 2, 4, 16)
    assert len(result) == 4
    assert result == expected_order[8:12]

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 3, 4, 16)
    assert len(result) == 4
    assert result == expected_order[12:16]

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 0, 3, 16)
    assert len(result) == 6
    assert result == expected_order[0:6]

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 2, 3, 16)
    assert len(result) == 5
    assert result == expected_order[11:16]

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 0, 1, 16)
    assert len(result) == 16
    assert result == expected_order

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 9, 10, 16)
    assert len(result) == 1
    assert result == [expected_order[15]]


def test_extended_get_trigger_samples():
    TriggerStorageCPP(TMP_DIR).save_trigger_sample(
        1,
        2,
        3,
        np.array([(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)], dtype=np.dtype("i8,f8")),
        4,
    )
    TriggerStorageCPP(TMP_DIR).save_trigger_sample(
        1,
        2,
        3,
        np.array([(3, 3.0), (4, 4.0), (5, 5.0), (6, 6.0)], dtype=np.dtype("i8,f8")),
        5,
    )

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

    TriggerStorageCPP(TMP_DIR).save_trigger_sample(
        1,
        2,
        3,
        np.array(
            [(7, 7.0), (8, 8.0), (9, 9.0), (10, 10.0), (11, 11.0)],
            dtype=np.dtype("i8,f8"),
        ),
        6,
    )
    expected_order = expected_order + [
        (7, 7.0),
        (8, 8.0),
        (9, 9.0),
        (10, 10.0),
        (11, 11.0),
    ]

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 0, 2, 12)
    assert len(result) == 6
    assert result == expected_order[0:6]

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 1, 3, 12)
    assert len(result) == 4
    assert result == expected_order[4:8]

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 2, 3, 12)
    assert len(result) == 4
    assert result == expected_order[8:12]

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 0, 12, 12)
    assert len(result) == 1
    assert result == [expected_order[0]]

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 1, 5, 12)
    assert len(result) == 3
    assert result == expected_order[3:6]

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 2, 5, 12)
    assert len(result) == 2
    assert result == expected_order[6:8]

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 3, 5, 12)
    assert len(result) == 2
    assert result == expected_order[8:10]

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 4, 5, 12)
    assert len(result) == 2
    assert result == expected_order[10:12]

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, -1, -1, 13)
    assert len(result) == 13
    assert result == expected_order

    TriggerStorageCPP(TMP_DIR).save_trigger_sample(
        1,
        2,
        30,
        np.array([(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)], dtype=np.dtype("i8,f8")),
        6,
    )

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 0, 2, 13)
    assert len(result) == 7
    assert result == expected_order[0:7]

    result = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 30, -1, -1, 4)
    assert len(result) == 4
    assert result == expected_order[0:4]


def test_get_trigger_samples_no_file():
    tmp = TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3)
    assert not tmp


def test_get_trigger_samples_illegal_workers():
    with pytest.raises(AssertionError):
        TriggerStorageCPP(TMP_DIR).get_trigger_samples(1, 2, 3, 0, -1, 2)


def test_init_directory():
    os.rmdir(TMP_DIR)
    _ = TriggerStorageCPP(TMP_DIR)

    assert os.path.exists(TMP_DIR)
