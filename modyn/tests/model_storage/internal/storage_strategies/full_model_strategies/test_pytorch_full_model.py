import pathlib
import tempfile
from zipfile import ZIP_DEFLATED

import torch
from modyn.model_storage.internal.storage_strategies.full_model_strategies import PyTorchFullModel
from modyn.utils import unzip_file, zip_file


def test_store_model():
    full_model_strategy = PyTorchFullModel(
        zipping_dir=pathlib.Path(), zip_activated=False, zip_algorithm_name="", config={}
    )
    with tempfile.NamedTemporaryFile() as temporary_file:
        temp_file_path = pathlib.Path(temporary_file.name)

        full_model_strategy.store_model({"conv_1": True}, temp_file_path)

        loaded_state = torch.load(temp_file_path)

        assert loaded_state["conv_1"]


def test_store_model_zipped():
    full_model_strategy = PyTorchFullModel(
        zipping_dir=pathlib.Path(), zip_activated=True, zip_algorithm_name="ZIP_DEFLATED", config={}
    )
    with tempfile.TemporaryDirectory() as temp_directory:
        directory_path = pathlib.Path(temp_directory)

        zipped_file_path = directory_path / "zipped.model"
        full_model_strategy.store_model({"conv_1": True}, zipped_file_path)

        unzipped_file_path = pathlib.Path(directory_path / "unzipped.model")
        unzip_file(zipped_file_path, unzipped_file_path, compression=ZIP_DEFLATED)

        loaded_state = torch.load(unzipped_file_path)
        assert loaded_state["conv_1"]


def test_load_model():
    full_model_strategy = PyTorchFullModel(
        zipping_dir=pathlib.Path(), zip_activated=False, zip_algorithm_name="", config={}
    )
    with tempfile.NamedTemporaryFile() as temporary_file:
        temp_file_path = pathlib.Path(temporary_file.name)

        torch.save({"conv_1": True}, temp_file_path)

        state_dict = {"conv_1": False}
        full_model_strategy.load_model(state_dict, temp_file_path)

        assert state_dict["conv_1"]


def test_load_model_zipped():
    with tempfile.TemporaryDirectory() as temp_directory:
        directory_path = pathlib.Path(temp_directory)
        full_model_strategy = PyTorchFullModel(
            zipping_dir=directory_path, zip_activated=True, zip_algorithm_name="ZIP_DEFLATED", config={}
        )

        model_path = directory_path / "basic.model"
        torch.save({"conv_1": True}, model_path)
        zipped_model_path = directory_path / "zipped.model"
        zip_file(model_path, zipped_model_path, compression=ZIP_DEFLATED)

        state_dict = {"conv_1": False}
        full_model_strategy.load_model(state_dict, zipped_model_path)

        assert state_dict["conv_1"]


def test_store_then_load():
    with tempfile.NamedTemporaryFile() as temporary_file:
        temp_file_path = pathlib.Path(temporary_file.name)
        full_model_strategy = PyTorchFullModel(
            zipping_dir=temp_file_path.parent, zip_activated=True, zip_algorithm_name="ZIP_DEFLATED", config={}
        )

        model_state = {"conv_1": True}
        full_model_strategy.store_model(model_state, temp_file_path)
        loaded_state = {"conv_1": False}
        full_model_strategy.load_model(loaded_state, temp_file_path)

        assert loaded_state["conv_1"]
