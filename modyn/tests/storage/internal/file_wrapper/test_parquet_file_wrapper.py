import os

import pytest

from modyn.storage.internal.file_wrapper.parquet_file_wrapper import ParquetFileWrapper


test_dir = os.getcwd() + os.path.sep + os.path.join('test_tmp', 'modyn', 'mnist')
file_path = os.getcwd() + os.path.sep + os.path.join('test_tmp', 'modyn', 'mnist', 'test.parquet')


def test_init():
    file_wrapper = ParquetFileWrapper(file_path)
    assert file_wrapper.file_path == file_path


def test_get_size():
    with pytest.raises(NotImplementedError):
        file_wrapper = ParquetFileWrapper(file_path)
        assert file_wrapper.get_size() == 10000


def test_get_samples():
    with pytest.raises(NotImplementedError):
        file_wrapper = ParquetFileWrapper(file_path)
        _ = file_wrapper.get_samples(0, 1)


def test_get_sample():
    with pytest.raises(NotImplementedError):
        file_wrapper = ParquetFileWrapper(file_path)
        _ = file_wrapper.get_sample(0)


def test_get_samples_from_indices():
    with pytest.raises(NotImplementedError):
        file_wrapper = ParquetFileWrapper(file_path)
        _ = file_wrapper.get_samples_from_indices([0, 1])
