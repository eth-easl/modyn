import os
import pathlib
import shutil

import pytest
from modyn.storage.internal.file_wrapper.binary_file_wrapper import BinaryFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType

TMP_DIR = str(pathlib.Path(os.path.abspath(__file__)).parent / "test_tmp" / "modyn")
FILE_PATH = str(pathlib.Path(os.path.abspath(__file__)).parent / "test_tmp" / "modyn" / "test.bin")
FILE_DATA = b"\x00\x01\x00\x02\x00\x01\x00\x0f\x00\x00\x07\xd0"  # [1,2,1,15,0,2000]
INVALID_FILE_EXTENSION_PATH = str(pathlib.Path(os.path.abspath(__file__)).parent / "test_tmp" / "modyn" / "test.txt")
FILE_WRAPPER_CONFIG = {
    "record_size": 4,
    "label_size": 2,
    "byteorder": "big",
}
SMALL_RECORD_SIZE_CONFIG = {
    "record_size": 2,
    "label_size": 2,
    "byteorder": "big",
}
INDIVISIBLE_RECORD_SIZE_CONFIG = {
    "record_size": 5,
    "label_size": 2,
    "byteorder": "big",
}


def setup():
    os.makedirs(TMP_DIR, exist_ok=True)

    with open(FILE_PATH, "wb") as file:
        file.write(FILE_DATA)


def teardown():
    os.remove(FILE_PATH)
    shutil.rmtree(TMP_DIR)


class MockFileSystemWrapper:
    def __init__(self, file_path):
        self.file_path = file_path

    def get(self, file_path):
        with open(file_path, "rb") as file:
            return file.read()

    def get_size(self, path):
        return os.path.getsize(path)


def test_init():
    file_wrapper = BinaryFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    assert file_wrapper.file_path == FILE_PATH
    assert file_wrapper.file_wrapper_type == FileWrapperType.BinaryFileWrapper


def test_init_with_small_record_size_config():
    with pytest.raises(ValueError):
        BinaryFileWrapper(FILE_PATH, SMALL_RECORD_SIZE_CONFIG, MockFileSystemWrapper(FILE_PATH))


def test_init_with_invalid_file_extension():
    with pytest.raises(ValueError):
        BinaryFileWrapper(
            INVALID_FILE_EXTENSION_PATH,
            FILE_WRAPPER_CONFIG,
            MockFileSystemWrapper(INVALID_FILE_EXTENSION_PATH),
        )


def test_init_with_indivisiable_record_size():
    with pytest.raises(ValueError):
        BinaryFileWrapper(
            FILE_PATH,
            INDIVISIBLE_RECORD_SIZE_CONFIG,
            MockFileSystemWrapper(FILE_PATH),
        )


def test_get_number_of_samples():
    file_wrapper = BinaryFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    assert file_wrapper.get_number_of_samples() == 3


def test_get_sample():
    file_wrapper = BinaryFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    sample = file_wrapper.get_sample(0)
    assert sample == b"\x00\x02"

    sample = file_wrapper.get_sample(2)
    assert sample == b"\x07\xd0"


def test_get_sample_with_invalid_index():
    file_wrapper = BinaryFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    with pytest.raises(IndexError):
        file_wrapper.get_sample(10)


def test_get_label():
    file_wrapper = BinaryFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    label = file_wrapper.get_label(0)
    assert label == 1

    label = file_wrapper.get_label(2)
    assert label == 0


def test_get_label_with_invalid_index():
    file_wrapper = BinaryFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    with pytest.raises(IndexError):
        file_wrapper.get_label(10)


def test_get_samples():
    file_wrapper = BinaryFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    samples = file_wrapper.get_samples(0, 1)
    assert len(samples) == 1
    assert samples[0] == b"\x00\x02"

    samples = file_wrapper.get_samples(0, 2)
    assert len(samples) == 2
    assert samples[0] == b"\x00\x02"
    assert samples[1] == b"\x00\x0f"


def test_get_samples_with_invalid_index():
    file_wrapper = BinaryFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    with pytest.raises(IndexError):
        file_wrapper.get_samples(0, 5)

    with pytest.raises(IndexError):
        file_wrapper.get_samples(3, 4)


def test_get_samples_from_indices():
    file_wrapper = BinaryFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    samples = file_wrapper.get_samples_from_indices([0, 2])
    assert len(samples) == 2
    assert samples[0] == b"\x00\x02"
    assert samples[1] == b"\x07\xd0"


def test_get_samples_from_indices_with_invalid_indices():
    file_wrapper = BinaryFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    with pytest.raises(IndexError):
        file_wrapper.get_samples_from_indices([-2, 1])
