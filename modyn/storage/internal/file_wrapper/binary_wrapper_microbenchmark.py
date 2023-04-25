from modyn.storage.internal.file_wrapper.binary_file_wrapper import BinaryFileWrapper
from modyn.storage.internal.file_wrapper.binary_file_wrapper_new import BinaryFileWrapperNew
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType
import os

FILE_PATH = "data.bin"
FILE_WRAPPER_CONFIG = {
    "record_size": 8,
    "label_size": 4,
    "byteorder": "little",
}


class MockFileSystemWrapper:
    def __init__(self, file_path):
        self.file_path = file_path
        self.filesystem_wrapper_type = "MockFileSystemWrapper"

    def get(self, file_path):
        with open(file_path, "rb") as file:
            return file.read()

    def get_size(self, path):
        return os.path.getsize(path)

def test_init():
    file_wrapper = BinaryFileWrapper(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))
    assert file_wrapper.file_path == FILE_PATH
    assert file_wrapper.file_wrapper_type == FileWrapperType.BinaryFileWrapper

    file_wrapper_new_non_native = BinaryFileWrapperNew(FILE_PATH, FILE_WRAPPER_CONFIG, MockFileSystemWrapper(FILE_PATH))

    mock_file_system_wrapper = MockFileSystemWrapper(FILE_PATH)
    mock_file_system_wrapper.filesystem_wrapper_type = FilesystemWrapperType.LocalFilesystemWrapper
    file_wrapper_new_native = BinaryFileWrapperNew(FILE_PATH, FILE_WRAPPER_CONFIG, mock_file_system_wrapper)

    return file_wrapper, file_wrapper_new_non_native, file_wrapper_new_native

def run():
    file_wrapper, file_wrapper_new_non_native, file_wrapper_new_native = test_init()
    print("Running benchmark for BinaryFileWrapper")
    labels = file_wrapper.get_all_labels()
    print("Running benchmark for BinaryFileWrapperNew (non-native)")
    labels_new_non_native = file_wrapper_new_non_native.get_all_labels()
    print("Running benchmark for BinaryFileWrapperNew (native)")
    labels_new_native = file_wrapper_new_native.get_all_labels()

    assert labels == labels_new_non_native
    assert labels == labels_new_native


if __name__ == "__main__":
    import random; 
    import struct; 
    encoded_integers = b''.join(struct.pack('<I', random.randint(0, 2147483647)) for _ in range(2*int(10)))
    padding = b'\x00' * ((2 * int(10) * 4) - len(encoded_integers))
    encoded_data = encoded_integers + padding
    open('data.bin', 'wb').write(encoded_data)
    run()