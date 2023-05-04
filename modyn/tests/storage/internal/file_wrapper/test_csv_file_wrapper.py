import os
import pathlib
import shutil
from io import StringIO

import pytest
from modyn.storage.internal.file_wrapper.csv_file_wrapper import CSVFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType

TMP_DIR = str(pathlib.Path(os.path.abspath(__file__)).parent / "test_tmp" / "modyn")
FILE_PATH = str(pathlib.Path(os.path.abspath(__file__)).parent / "test_tmp" / "modyn" / "test.csv")
FILE_DATA = (
    "hero,description,can_fly\nbatman,uses technology,0\nsuperman,flies through the air,1\nspiderman,"
    "uses a web,0\nwonder women,truth lasso,0 "
)
INVALID_FILE_EXTENSION_PATH = str(pathlib.Path(os.path.abspath(__file__)).parent / "test_tmp" / "modyn" / "test.txt")
# Try 2 configs
FILE_WRAPPER_CONFIG = {
    "number_cols": 2,
    "delimiter": ",",
    "file_extension": ".csv",
    "header": 0,
    "label_col_index": 0,
    "encoding": "utf-8",
    "labels": "can_fly",
}


def setup():
    os.makedirs(TMP_DIR, exist_ok=True)
    file = StringIO(FILE_DATA)

    with open(FILE_PATH, "w", encoding=FILE_WRAPPER_CONFIG.get("encoding")) as file_pointer:
        file.seek(0)
        shutil.copyfileobj(file, file_pointer)


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
    file_wrapper = CSVFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    assert file_wrapper.file_path == FILE_PATH
    assert file_wrapper.file_wrapper_type == FileWrapperType.CSVFileWrapper


def test_init_with_invalid_file_extension():
    with pytest.raises(ValueError):
        CSVFileWrapper(
            INVALID_FILE_EXTENSION_PATH,
            FILE_WRAPPER_CONFIG,
            MockFileSystemWrapper(INVALID_FILE_EXTENSION_PATH),
        )


def test_get_number_of_samples():
    file_wrapper = CSVFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    assert file_wrapper.get_number_of_samples() == 4


def test_get_sample():
    file_wrapper = CSVFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    sample = file_wrapper.get_sample(1)
    assert sample == b"hero,description\nsuperman,flies through the air\n"


def test_get_sample_with_invalid_index():
    file_wrapper = CSVFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    with pytest.raises(IndexError):
        file_wrapper.get_sample(10)


def test_get_label():
    file_wrapper = CSVFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    label = file_wrapper.get_label(1)
    assert label == 1


def test_get_all_labels():
    file_wrapper = CSVFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    assert file_wrapper.get_all_labels() == [0, 1, 0, 0]


def test_get_label_with_invalid_index():
    file_wrapper = CSVFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    with pytest.raises(IndexError):
        file_wrapper.get_label(10)


def test_get_samples():
    file_wrapper = CSVFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    samples = file_wrapper.get_samples(1, 3)
    print(samples)
    assert samples == b"hero,description\nsuperman,flies through the air\nspiderman,uses a web\n"


def test_get_samples_with_invalid_index():
    file_wrapper = CSVFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    with pytest.raises(IndexError):
        file_wrapper.get_samples(0, 5)

    with pytest.raises(IndexError):
        file_wrapper.get_samples(3, 6)


def test_get_samples_from_indices():
    file_wrapper = CSVFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    samples = file_wrapper.get_samples_from_indices([0, 1])
    # assert len(samples) == 2
    assert samples == b"hero,description\nbatman,uses technology\nsuperman,flies through the air\n"


def test_get_samples_from_indices_with_invalid_indices():
    file_wrapper = CSVFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    with pytest.raises(IndexError):
        file_wrapper.get_samples_from_indices([-2, 10])
