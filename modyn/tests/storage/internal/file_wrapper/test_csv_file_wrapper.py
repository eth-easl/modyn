import os
import pathlib
import shutil

import pytest
from modyn.storage.internal.file_wrapper.csv_file_wrapper import CsvFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType

TMP_DIR = str(pathlib.Path(os.path.abspath(__file__)).parent / "test_tmp" / "modyn")
FILE_PATH = str(pathlib.Path(os.path.abspath(__file__)).parent / "test_tmp" / "modyn" / "test.csv")
FILE_DATA = b"a;b;c;d;12\ne;f;g;h;76"
INVALID_FILE_EXTENSION_PATH = str(pathlib.Path(os.path.abspath(__file__)).parent / "test_tmp" / "modyn" / "test.txt")
FILE_WRAPPER_CONFIG = {
    "ignore_first_line": False,
    "label_index": 4,
    "separator": ";",
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
    file_wrapper = CsvFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    assert file_wrapper.file_path == FILE_PATH
    assert file_wrapper.file_wrapper_type == FileWrapperType.CsvFileWrapper
    assert file_wrapper.encoding == "utf-8"
    assert file_wrapper.label_index == 4
    assert not file_wrapper.ignore_first_line
    assert file_wrapper.separator == ";"


def test_init_with_invalid_file_extension():
    with pytest.raises(ValueError):
        CsvFileWrapper(
            INVALID_FILE_EXTENSION_PATH,
            FILE_WRAPPER_CONFIG,
            MockFileSystemWrapper(INVALID_FILE_EXTENSION_PATH),
        )


def test_get_number_of_samples():
    file_wrapper = CsvFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    assert file_wrapper.get_number_of_samples() == 2

    # check if the first line is correctly ignored
    file_wrapper.ignore_first_line = True
    assert file_wrapper.get_number_of_samples() == 1


def test_get_sample():
    file_wrapper = CsvFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    sample = file_wrapper.get_sample(0)
    assert sample == b"a;b;c;d"

    sample = file_wrapper.get_sample(1)
    assert sample == b"e;f;g;h"


def test_get_sample_with_invalid_index():
    file_wrapper = CsvFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    with pytest.raises(IndexError):
        file_wrapper.get_sample(10)


def test_get_label():
    file_wrapper = CsvFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    label = file_wrapper.get_label(0)
    assert label == 12

    label = file_wrapper.get_label(1)
    assert label == 76

    with pytest.raises(IndexError):
        file_wrapper.get_label(2)


def test_get_all_labels():
    file_wrapper = CsvFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    assert file_wrapper.get_all_labels() == [12, 76]


def test_get_label_with_invalid_index():
    file_wrapper = CsvFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    with pytest.raises(IndexError):
        file_wrapper.get_label(10)


def test_get_samples():
    file_wrapper = CsvFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    samples = file_wrapper.get_samples(0, 1)
    assert len(samples) == 1
    assert samples[0] == b"a;b;c;d"

    samples = file_wrapper.get_samples(0, 2)
    assert len(samples) == 2
    assert samples[0] == b"a;b;c;d"
    assert samples[1] == b"e;f;g;h"


def test_get_samples_with_invalid_index():
    file_wrapper = CsvFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    with pytest.raises(IndexError):
        file_wrapper.get_samples(0, 5)

    with pytest.raises(IndexError):
        file_wrapper.get_samples(3, 4)


def test_get_samples_from_indices_with_invalid_indices():
    file_wrapper = CsvFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    with pytest.raises(IndexError):
        file_wrapper.get_samples_from_indices([-2, 1])
