import json
import os
import pathlib
import pickle
import shutil

import webdataset as wds
from modyn.storage.internal.file_wrapper.webdataset_file_wrapper import WebdatasetFileWrapper
from PIL import Image

TEST_DIR = pathlib.Path(os.path.abspath(__file__)).parent / "test_tmp" / "modyn"
FILE_PATH = str(pathlib.Path(os.path.abspath(__file__)).parent / "test_tmp" / "modyn" / "test.tar")


def setup():
    width = 28
    height = 28
    os.makedirs(TEST_DIR, exist_ok=True)

    test_file = wds.TarWriter(FILE_PATH)
    for i in range(150):

        test_jpg = Image.new("RGB", (width, height), color=(255, 255, 255))
        for m in range(test_jpg.size[0]):  # pylint: disable=invalid-name
            for n in range(test_jpg.size[1]):  # pylint: disable=invalid-name
                if m * 28 + n == i:
                    test_jpg.putpixel((m, n), (0, 0, 0))
        test_json = json.dumps({"__key__": str(i), "id": i})
        test_file.write({"__key__": str(i), "jpg": test_jpg, "json": test_json, "cls": str(i)})
    test_file.close()


def teardown():
    os.remove(FILE_PATH)
    shutil.rmtree(TEST_DIR)


def test_init():
    file_wrapper = WebdatasetFileWrapper(FILE_PATH, {}, None)
    assert file_wrapper.file_path == FILE_PATH


def test_get_number_of_samples():
    file_wrapper = WebdatasetFileWrapper(FILE_PATH, {}, None)
    assert file_wrapper.get_number_of_samples() == 150


def test_get_samples():
    file_wrapper = WebdatasetFileWrapper(FILE_PATH, {}, None)
    samples = file_wrapper.get_samples(0, 1)

    samples = pickle.loads(samples)

    for i, sample in enumerate(samples):
        assert sample[0].shape == (28, 28, 3)
        m = i % 28  # pylint: disable=invalid-name
        n = i // 28  # pylint: disable=invalid-name
        assert sample[0][m, n].tolist() == [0, 0, 0]
        assert sample[1] == i
        assert sample[2]["id"] == i

    samples = file_wrapper.get_samples(100, 102)

    samples = pickle.loads(samples)

    for i, sample in enumerate(samples):
        assert sample[0].shape == (28, 28, 3)
        m = (i + 100) % 28  # pylint: disable=invalid-name
        n = (i + 100) // 28  # pylint: disable=invalid-name
        assert sample[0][m, n].tolist() == [0, 0, 0]
        assert sample[1] == i + 100
        assert sample[2]["id"] == i + 100


def test_get_sample():
    file_wrapper = WebdatasetFileWrapper(FILE_PATH, {}, None)
    sample = file_wrapper.get_sample(0)

    sample = pickle.loads(sample)

    for image, label, json_file in sample:
        assert image.shape == (28, 28, 3)
        m = label % 28  # pylint: disable=invalid-name
        n = label // 28  # pylint: disable=invalid-name
        assert image[m, n].tolist() == [0, 0, 0]
        assert json_file["id"] == label

    sample = file_wrapper.get_sample(100)

    sample = pickle.loads(sample)

    for image, label, json_file in sample:
        assert image.shape == (28, 28, 3)
        m = label % 28  # pylint: disable=invalid-name
        n = label // 28  # pylint: disable=invalid-name
        assert image[m, n].tolist() == [0, 0, 0]
        assert json_file["id"] == label


def test_get_label():
    file_wrapper = WebdatasetFileWrapper(FILE_PATH, {}, None)
    label = file_wrapper.get_label(0)

    assert label == 0

    label = file_wrapper.get_label(100)

    assert label == 100


def test_get_samples_from_indices():
    file_wrapper = WebdatasetFileWrapper(FILE_PATH, {}, None)

    indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    samples = file_wrapper.get_samples_from_indices(indices)

    sample = pickle.loads(samples)

    i = 0
    for image, label, json_file in sample:
        assert image.shape == (28, 28, 3)
        m = indices[i] % 28  # pylint: disable=invalid-name
        n = indices[i] // 28  # pylint: disable=invalid-name
        assert image[m, n].tolist() == [0, 0, 0]
        assert label == indices[i]
        assert json_file["id"] == label
        i += 1

    assert i == 10

    indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    samples = file_wrapper.get_samples_from_indices(indices)

    sample = pickle.loads(samples)

    i = 0
    for image, label, json_file in sample:
        assert image.shape == (28, 28, 3)
        m = indices[i] % 28  # pylint: disable=invalid-name
        n = indices[i] // 28  # pylint: disable=invalid-name
        assert image[m, n].tolist() == [0, 0, 0]
        assert label == indices[i]
        assert json_file["id"] == label
        i += 1

    assert i == 10

    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    samples = file_wrapper.get_samples_from_indices(indices)

    sample = pickle.loads(samples)

    i = 0
    for image, label, json_file in sample:
        assert image.shape == (28, 28, 3)
        m = indices[i] % 28  # pylint: disable=invalid-name
        n = indices[i] // 28  # pylint: disable=invalid-name
        assert image[m, n].tolist() == [0, 0, 0]
        assert label == indices[i]
        assert json_file["id"] == label
        i += 1

    assert i == 10
