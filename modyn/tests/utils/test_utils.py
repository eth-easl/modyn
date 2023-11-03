# pylint: disable=unused-argument,redefined-outer-name
import io
import pathlib
import tempfile
from unittest.mock import patch

import grpc
import numpy as np
import pytest
import torch
import yaml
from modyn.common.trigger_storage_cpp import TriggerSampleStorage
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.trainer_server.internal.trainer.remote_downsamplers import RemoteLossDownsampling
from modyn.utils import (
    calculate_checksum,
    convert_timestr_to_seconds,
    current_time_millis,
    deserialize_function,
    dynamic_module_import,
    flatten,
    get_partition_for_worker,
    get_tensor_byte_size,
    grpc_connection_established,
    instantiate_class,
    model_available,
    package_available_and_can_be_imported,
    reconstruct_tensor_from_bytes,
    seed_everything,
    trigger_available,
    unzip_file,
    validate_timestr,
    validate_yaml,
    zip_file,
)


@patch.object(GRPCHandler, "init_storage", lambda self: None)
def test_connection_established_times_out():
    assert not grpc_connection_established(grpc.insecure_channel("1.2.3.4:42"), 0.5)


@patch("grpc.channel_ready_future")
def test_connection_established_works_mocked(test_channel_ready_future):
    # Pretty dumb test, needs E2E test with running server.

    class MockFuture:
        def result(self, timeout):
            return True

    test_channel_ready_future.return_value = MockFuture()
    assert grpc_connection_established(grpc.insecure_channel("1.2.3.4:42"))


def test_dynamic_module_import():
    assert dynamic_module_import("modyn.utils") is not None


def test_model_available():
    assert model_available("ResNet18")
    assert not model_available("NonExistingModel")


def test_trigger_available():
    assert trigger_available("TimeTrigger")
    assert not trigger_available("NonExistingTrigger")


def test_validate_yaml():
    with open(
        pathlib.Path("modyn") / "config" / "examples" / "modyn_config.yaml",
        "r",
        encoding="utf-8",
    ) as concrete_file:
        file = yaml.safe_load(concrete_file)
    assert validate_yaml(file, pathlib.Path("modyn") / "config" / "schema" / "modyn_config_schema.yaml")[0]
    with open(
        pathlib.Path("modyn") / "config" / "examples" / "example-pipeline.yaml",
        "r",
        encoding="utf-8",
    ) as concrete_file:
        file = yaml.safe_load(concrete_file)
    assert validate_yaml(file, pathlib.Path("modyn") / "config" / "schema" / "pipeline-schema.yaml")[0]
    assert not validate_yaml({}, pathlib.Path("modyn") / "config" / "schema" / "modyn_config_schema.yaml")[0]


@patch("time.time", lambda: 0.4)
def test_current_time_millis():
    assert current_time_millis() == 400


def test_validate_timestr():
    assert validate_timestr("10s")
    assert validate_timestr("10m")
    assert validate_timestr("10h")
    assert validate_timestr("10d")
    assert not validate_timestr("10")
    assert not validate_timestr("10x")
    assert not validate_timestr("10s10m")


def test_convert_timestr_to_seconds():
    assert convert_timestr_to_seconds("10s") == 10
    assert convert_timestr_to_seconds("10m") == 600
    assert convert_timestr_to_seconds("10h") == 36000
    assert convert_timestr_to_seconds("10d") == 864000


def test_package_available_and_can_be_imported():
    assert package_available_and_can_be_imported("modyn")
    assert not package_available_and_can_be_imported("testpackage")


def test_flatten():
    assert flatten([[1, 2, 3, 4]]) == [1, 2, 3, 4]
    assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert flatten([[1, 2], [3, 4], [5, 6]]) == [1, 2, 3, 4, 5, 6]


def test_get_partition_for_worker():
    samples = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    assert get_partition_for_worker(0, 3, len(samples)) == (0, 4)
    assert get_partition_for_worker(1, 3, len(samples)) == (4, 3)
    assert get_partition_for_worker(2, 3, len(samples)) == (7, 3)

    with pytest.raises(ValueError):
        get_partition_for_worker(3, 3, len(samples))
    with pytest.raises(ValueError):
        get_partition_for_worker(-1, 3, len(samples))

    assert get_partition_for_worker(0, 2, len(samples)) == (0, 5)

    samples = [1, 2, 3]
    assert get_partition_for_worker(0, 8, len(samples)) == (0, 1)
    assert get_partition_for_worker(1, 8, len(samples)) == (1, 1)
    assert get_partition_for_worker(2, 8, len(samples)) == (2, 1)
    assert get_partition_for_worker(3, 8, len(samples)) == (0, 0)
    assert get_partition_for_worker(4, 8, len(samples)) == (0, 0)
    assert get_partition_for_worker(5, 8, len(samples)) == (0, 0)
    assert get_partition_for_worker(6, 8, len(samples)) == (0, 0)
    assert get_partition_for_worker(7, 8, len(samples)) == (0, 0)


def test_deserialize_function():
    test_func = "def test_func(x: int, y: int) -> int:\n\treturn x + y"
    assert deserialize_function(test_func, "test_func")(5, 3) == 8

    test_func_import = "import torch\ndef test_func(x: torch.Tensor):\n\treturn x * 3"

    assert torch.all(
        torch.eq(
            deserialize_function(test_func_import, "test_func")(torch.ones(4)),
            torch.ones(4) * 3,
        )
    )


def test_deserialize_function_invalid():
    test_func = "def test_func():\n\treturn 0"

    with pytest.raises(ValueError):
        deserialize_function(test_func, "test")
    deserialized_test_func = deserialize_function(test_func, "test_func")
    assert deserialized_test_func() == 0

    invalid_func = "test_func=1"
    with pytest.raises(ValueError):
        deserialize_function(invalid_func, "test_func")

    empty_func = ""
    assert deserialize_function(empty_func, "test_func") is None


def test_seed():
    seed_everything(12)
    torch_master = torch.randn(10)
    np_master = np.random.randn(10)

    seed_everything(67)
    assert not np.all(np.equal(np_master, np.random.randn(10)))
    assert not torch.equal(torch_master, torch.randn(10))

    for _ in range(23):
        seed_everything(12)
        assert torch.equal(torch_master, torch.randn(10))
        assert np.all(np.equal(np_master, np.random.randn(10)))


def test_instantiate_class_existing():
    # class with a single parameter
    trigger_storage = instantiate_class("modyn.common.trigger_storage_cpp", "TriggerSampleStorage", "test_path")
    assert isinstance(trigger_storage, TriggerSampleStorage)
    assert trigger_storage.trigger_sample_directory == "test_path"
    # class with several parameters
    remote_downsampler = instantiate_class(
        "modyn.trainer_server.internal.trainer.remote_downsamplers",
        "RemoteLossDownsampling",
        10,
        11,
        64,
        {"downsampling_ratio": 67},
        {},
        "cpu",
    )
    assert isinstance(remote_downsampler, RemoteLossDownsampling)
    assert remote_downsampler.downsampling_ratio == 67
    assert remote_downsampler.pipeline_id == 10
    assert remote_downsampler.trigger_id == 11


def test_instantiate_class_not_existing():
    # missing package
    with pytest.raises(ModuleNotFoundError):
        instantiate_class("modyn.common.rumble_db_storage", "RumbleStorage", "test_path")
    # missing class
    with pytest.raises(ValueError):
        instantiate_class("modyn.common.trigger_sample", "BeautifulAmazingStorage", "test_path")
    # missing parameters
    with pytest.raises(TypeError):
        instantiate_class("modyn.common.trigger_sample", "TriggerSampleStorage")


def test_calculate_checksum():
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir_path = pathlib.Path(tempdir)

        with open(tempdir_path / "testfile1.txt", "w", encoding="utf-8") as file:
            file.write("This is a test")

        with open(tempdir_path / "testfile2.txt", "w", encoding="utf-8") as file:
            file.write("This is a test")

        assert calculate_checksum(tempdir_path / "testfile1.txt") == calculate_checksum(tempdir_path / "testfile2.txt")
        assert calculate_checksum(tempdir_path / "testfile1.txt", chunk_num_blocks=20) == calculate_checksum(
            tempdir_path / "testfile2.txt", chunk_num_blocks=10
        )
        assert calculate_checksum(tempdir_path / "testfile1.txt", hash_func_name="blake2s") != calculate_checksum(
            tempdir_path / "testfile2.txt", chunk_num_blocks=10
        )


def test_zip_and_unzip_file():
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir_path = pathlib.Path(tempdir)

        text_file_path = tempdir_path / "testfile.txt"
        zip_file_path = tempdir_path / "testfile.zip"

        with open(text_file_path, "w", encoding="utf-8") as file:
            file.write("This is a testfile!")

        zip_file(text_file_path, zip_file_path, remove_file=True)

        assert not text_file_path.exists()
        assert zip_file_path.exists() and zip_file_path.is_file()

        unzip_file(zip_file_path, text_file_path, remove_file=True)

        assert not zip_file_path.exists()
        assert text_file_path.exists() and text_file_path.is_file()

        with open(text_file_path, "r", encoding="utf-8") as file:
            assert file.read() == "This is a testfile!"


def test_read_tensor_from_bytes():
    buf = io.BytesIO()
    buf.write(b"\x01\x00\x00\x00")
    buf.write(b"\x02\x00\x00\x00")
    buf.write(b"\x03\x00\x00\x00")
    buf.write(b"\x04\x00\x00\x00")
    buf.seek(0)
    res = reconstruct_tensor_from_bytes(torch.ones((2, 2), dtype=torch.int32), buf.getvalue())

    assert res[0, 0] == 1 and res[0, 1] == 2 and res[1, 0] == 3 and res[1, 1] == 4

    buf.seek(0, io.SEEK_END)
    buf.write(b"\xff\x00\x00\x00")
    buf.write(b"\x0f\x00\x00\x00")
    buf.seek(0)

    res = reconstruct_tensor_from_bytes(torch.ones((1, 6), dtype=torch.int32), buf.getvalue())
    assert (
        res[0, 0] == 1 and res[0, 1] == 2 and res[0, 2] == 3 and res[0, 3] == 4 and res[0, 4] == 255 and res[0, 5] == 15
    )

    buf_floats = io.BytesIO()
    buf_floats.write(b"\x00\x00\x00\x3f")
    buf_floats.write(b"\x00\x00\x00\x3e")

    res = reconstruct_tensor_from_bytes(torch.ones((2, 1), dtype=torch.float32), buf_floats.getvalue())
    assert res[0, 0] == 1.0 / 2 and res[1, 0] == 1.0 / 8


def test_get_tensor_byte_size():
    tensor = torch.ones((3, 3, 3), dtype=torch.int32)
    assert get_tensor_byte_size(tensor) == 3 * 3 * 3 * 4

    tensor = torch.ones((5, 5), dtype=torch.float64) * 5
    assert get_tensor_byte_size(tensor) == 5 * 5 * 8

    tensor = torch.ones(10, dtype=torch.float32)
    assert get_tensor_byte_size(tensor) == 40
