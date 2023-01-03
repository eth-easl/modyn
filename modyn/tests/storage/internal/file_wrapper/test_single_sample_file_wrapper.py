import pathlib
import os
import shutil
import pytest

from modyn.storage.internal.file_wrapper.single_sample_file_wrapper import SingleSampleFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType


FILE_PATH = pathlib.Path(os.path.abspath(__file__)).parent / 'test_tmp' / 'modyn' / 'test.tar'


def setup():
    os.makedirs(FILE_PATH.parent, exist_ok=True)
    with open(FILE_PATH, 'w', encoding='utf-8') as file:
        file.write('test')


def teardown():
    os.remove(FILE_PATH)
    shutil.rmtree(FILE_PATH.parent)


def test_init():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH)
    assert file_wrapper.file_path == FILE_PATH
    assert file_wrapper.file_wrapper_type == FileWrapperType.SingleSampleFileWrapper


def test_get_size():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH)
    assert file_wrapper.get_size() == 1


def test_get_samples():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH)
    samples = file_wrapper.get_samples(0, 1)
    assert samples == b'test'


def test_get_samples_with_invalid_indices():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH)
    with pytest.raises(IndexError):
        file_wrapper.get_samples(0, 2)


def test_get_sample():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH)
    sample = file_wrapper.get_sample(0)
    assert sample == b'test'


def test_get_sample_with_invalid_index():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH)
    with pytest.raises(IndexError):
        file_wrapper.get_sample(1)


def test_get_samples_from_indices():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH)
    samples = file_wrapper.get_samples_from_indices([0])
    assert samples == b'test'


def test_get_samples_from_indices_with_invalid_indices():
    file_wrapper = SingleSampleFileWrapper(FILE_PATH)
    with pytest.raises(IndexError):
        file_wrapper.get_samples_from_indices([0, 1])
