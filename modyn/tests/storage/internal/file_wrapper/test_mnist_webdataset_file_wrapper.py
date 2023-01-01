import os

import webdataset as wds

from modyn.storage.internal.file_wrapper.mnist_webdataset_file_wrapper import Mnist_webdatasetFileWrapper

file_path = os.path.sep + os.path.join('tmp', 'modyn', 'mnist', 'test')


def setup():
    os.makedirs(os.path.sep + os.path.join('tmp', 'modyn', 'mnist'), exist_ok=True)

    test_file = wds.WebDatasetWriter(file_path, maxcount=10000)
    for i in range(10000):
        test_jpg = b'0' + str(i).encode('utf-8')
        test_cls = b'0' + str(i).encode('utf-8') + b'0'
        test_file.write({'__key__': str(i), 'jpg': test_jpg, 'cls': test_cls})
    test_file.close()


def teardown():
    os.remove(file_path)
    os.remove(os.path.sep + os.path.join('tmp', 'modyn', 'mnist'))


def test_init():
    file_wrapper = Mnist_webdatasetFileWrapper(file_path)
    assert file_wrapper.file_path == 'test'


def test_get_size():
    file_wrapper = Mnist_webdatasetFileWrapper(file_path)
    assert file_wrapper.get_size() == 10000


def test_get_samples():
    file_wrapper = Mnist_webdatasetFileWrapper(file_path)
    samples = file_wrapper.get_samples(0, 1)
    assert samples[0][0] == b'0' + str(0).encode('utf-8')
    assert samples[1][0] == b'0' + str(0).encode('utf-8') + b'0'
    assert samples[0][1] == b'0' + str(1).encode('utf-8')
    assert samples[1][1] == b'0' + str(1).encode('utf-8') + b'0'

    samples = file_wrapper.get_samples(100, 102)
    assert samples[0][0] == b'0' + str(100).encode('utf-8')
    assert samples[1][0] == b'0' + str(100).encode('utf-8') + b'0'
    assert samples[0][1] == b'0' + str(101).encode('utf-8')
    assert samples[1][1] == b'0' + str(101).encode('utf-8') + b'0'
    assert samples[0][2] == b'0' + str(102).encode('utf-8')
    assert samples[1][2] == b'0' + str(102).encode('utf-8') + b'0'


def test_get_sample():
    file_wrapper = Mnist_webdatasetFileWrapper(file_path)
    sample = file_wrapper.get_sample(0)
    assert sample[0] == b'0' + str(0).encode('utf-8')
    assert sample[1] == b'0' + str(0).encode('utf-8') + b'0'

    sample = file_wrapper.get_sample(100)
    assert sample[0] == b'0' + str(100).encode('utf-8')
    assert sample[1] == b'0' + str(100).encode('utf-8') + b'0'
