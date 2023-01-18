import os
import pathlib
import shutil

import pytest
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.file_wrapper.single_sample_file_wrapper import SingleSampleFileWrapper

TMP_DIR = str(pathlib.Path(os.path.abspath(__file__)).parent / "test_tmp" / "modyn")
FILE_PATH = str(pathlib.Path(os.path.abspath(__file__)).parent / "test_tmp" / "modyn" / "test.png")
METADATA_PATH = str(pathlib.Path(os.path.abspath(__file__)).parent / "test_tmp" / "modyn" / "test.json")
FILE_WRAPPER_CONFIG = {"file_extension": ".png", "label_file_extension": ".json"}
FILE_WRAPPER_CONFIG_MIN = {"file_extension": ".png"}


def setup():
    os.makedirs(TMP_DIR, exist_ok=True)
    with open(FILE_PATH, "w", encoding="utf-8") as file:
        file.write("test")
    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        file.write('{"test": "test"}')


def teardown():
    os.remove(FILE_PATH)
    os.remove(METADATA_PATH)
    shutil.rmtree(TMP_DIR)


class MockFileSystemWrapper:
    def __init__(self, file_path):
        self.file_path = file_path

    def get(self, file_path):
        with open(file_path, "rb") as file:
            return file.read()


def test_init():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    assert file_wrapper.file_path == FILE_PATH
    assert file_wrapper.file_wrapper_type == FileWrapperType.SingleSampleFileWrapper


def test_get_number_of_samples():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    assert file_wrapper.get_number_of_samples() == 1


def test_get_samples():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    samples = file_wrapper.get_samples(0, 1)
    assert samples.startswith(b"test")


def test_get_samples_with_invalid_indices():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    with pytest.raises(IndexError):
        file_wrapper.get_samples(0, 2)


def test_get_sample():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    sample = file_wrapper.get_sample(0)
    assert sample.startswith(b"test")


def test_get_sample_with_invalid_index():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    with pytest.raises(IndexError):
        file_wrapper.get_sample(1)


def test_get_label():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    label = file_wrapper.get_label(0)
    assert label == b'{"test": "test"}'


def test_get_label_with_invalid_index():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    with pytest.raises(IndexError):
        file_wrapper.get_label(1)


def test_get_label_no_label():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG_MIN, MockFileSystemWrapper(FILE_PATH))
    label = file_wrapper.get_label(0)
    assert label is None


def test_get_samples_from_indices():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    samples = file_wrapper.get_samples_from_indices([0])
    assert samples.startswith(b"test")


def test_get_samples_from_indices_with_invalid_indices():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    with pytest.raises(IndexError):
        file_wrapper.get_samples_from_indices([0, 1])
