from modyn.storage.internal.file_wrapper.binary_file_wrapper import BinaryFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
import os

FILE_PATH = "/tmp/modyn_test/data.bin"
FILE_WRAPPER_CONFIG = {
    "record_size": 8,
    "label_size": 4,
    "byteorder": "big",
}


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
    return file_wrapper

def run():
    file_wrapper = test_init()
    file_wrapper.get_all_labels()
