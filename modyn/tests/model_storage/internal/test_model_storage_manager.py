# pylint: disable=unused-argument
import json
import os
import pathlib
import tempfile
from unittest.mock import MagicMock, patch
from zipfile import ZIP_DEFLATED

import torch
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import TrainedModel
from modyn.metadata_database.utils import ModelStorageStrategyConfig
from modyn.model_storage.internal import ModelStorageManager
from modyn.model_storage.internal.storage_strategies.full_model_strategies import PyTorchFullModel
from modyn.model_storage.internal.storage_strategies.incremental_model_strategies import WeightsDifference
from modyn.models import ResNet18
from modyn.utils import unzip_file, zip_file

DATABASE = pathlib.Path(os.path.abspath(__file__)).parent / "test_model_storage.database"


def get_modyn_config():
    return {
        "model_storage": {"port": "50051", "ftp_port": "5223"},
        "trainer_server": {"hostname": "localhost", "ftp_port": "5222"},
        "metadata_database": {
            "drivername": "sqlite",
            "username": "",
            "password": "",
            "host": "",
            "port": 0,
            "database": f"{DATABASE}",
        },
    }


def setup():
    if os.path.exists(DATABASE):
        os.remove(DATABASE)

    with MetadataDatabaseConnection(get_modyn_config()) as database:
        database.create_tables()

        full_model_strategy = ModelStorageStrategyConfig(name="PyTorchFullModel")
        inc_model_strategy = ModelStorageStrategyConfig(name="WeightsDifference")
        inc_model_strategy.zip = False
        inc_model_strategy.config = json.dumps({"operator": "sub"})
        database.register_pipeline(
            1, "ResNet18", json.dumps({"num_classes": 10}), True, full_model_strategy, inc_model_strategy, 5
        )


def teardown():
    os.remove(DATABASE)


class MockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, data):
        return data


def get_mock_model_after() -> MockModel:
    model_after = MockModel()
    model_after._weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32) * 3)

    return model_after


def test_init():
    manager = ModelStorageManager(get_modyn_config(), pathlib.Path("storage"))

    assert manager._modyn_config == get_modyn_config()
    assert manager._storage_dir == pathlib.Path("storage")


def test__get_previous_model():
    with MetadataDatabaseConnection(get_modyn_config()) as database:
        database.add_trained_model(10, 2, "model.modyn")

    manager = ModelStorageManager(get_modyn_config(), pathlib.Path("storage"))
    previous_model = manager._get_previous_model(10, 3)
    assert previous_model and previous_model.trigger_id == 2
    assert manager._get_previous_model(10, 2) is None


def test__get_base_model_state():
    manager = ModelStorageManager(get_modyn_config(), pathlib.Path("storage"))
    model_state = manager._get_base_model_state(1)

    assert len(model_state) == 122


def test__reconstruct_model():
    mock_model = MockModel()
    model_state = mock_model.state_dict()
    full_model_strategy = PyTorchFullModel(zip_activated=False, zip_algorithm_name="", config={})
    incremental_model_strategy = WeightsDifference(zip_activated=False, zip_algorithm_name="", config={})

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_directory_path = pathlib.Path(temp_dir)
        manager = ModelStorageManager(get_modyn_config(), temp_directory_path)

        prev_model_file_name = "before.model"
        full_model_strategy.save_model(model_state, temp_directory_path / prev_model_file_name)

        difference_model_file_name = "difference.model"
        incremental_model_strategy.save_model(
            get_mock_model_after().state_dict(), model_state, temp_directory_path / difference_model_file_name
        )

        with MetadataDatabaseConnection(get_modyn_config()) as database:
            prev_model_id = database.add_trained_model(15, 3, prev_model_file_name)
            curr_model_id = database.add_trained_model(15, 4, difference_model_file_name, parent_model=prev_model_id)

        manager._reconstruct_model(curr_model_id, model_state, manager.get_model_storage_strategy(1))

        assert model_state["_weight"].item() == 3  # pylint: disable=unsubscriptable-object


def test__handle_new_model_full():
    with MetadataDatabaseConnection(get_modyn_config()) as database:
        database.add_trained_model(1, 4, "model.modyn")

    mock_model = MockModel()
    model_state = mock_model.state_dict()

    manager = ModelStorageManager(get_modyn_config(), pathlib.Path("storage"))

    with tempfile.NamedTemporaryFile() as temporary_file:
        temp_file_path = pathlib.Path(temporary_file.name)

        parent_id = manager._handle_new_model(1, 5, model_state, temp_file_path, manager.get_model_storage_strategy(1))
        assert parent_id is None

        loaded_state = torch.load(temp_file_path)
        assert loaded_state["_weight"].item() == 1


@patch.object(ModelStorageManager, "_get_base_model_state", return_value=MockModel().state_dict())
@patch.object(ModelStorageManager, "_reconstruct_model")
@patch.object(ModelStorageManager, "_get_previous_model", return_value=TrainedModel(model_id=101))
def test__handle_new_model_incremental(
    previous_model_mock, reconstruct_model_mock: MagicMock, base_model_state_mock: MagicMock
):
    manager = ModelStorageManager(get_modyn_config(), pathlib.Path("storage"))

    with tempfile.NamedTemporaryFile() as temporary_file:
        temp_file_path = pathlib.Path(temporary_file.name)

        parent_id = manager._handle_new_model(
            5, 4, get_mock_model_after().state_dict(), temp_file_path, manager.get_model_storage_strategy(1)
        )

        assert parent_id == 101

        with open(temp_file_path, "rb") as model_file:
            assert model_file.read() == b"\x00\x00\x00\x40"

        base_model_state_mock.assert_called_once_with(5)
        previous_model_mock.assert_called_once_with(5, 4)


def test_get_model_storage_strategy():
    with MetadataDatabaseConnection(get_modyn_config()) as database:
        simple_pipeline = database.register_pipeline(
            74,
            "ResNet18",
            json.dumps({"num_classes": 10}),
            True,
            ModelStorageStrategyConfig(name="PyTorchFullModel"),
            None,
            None,
        )

        full_model_strategy = ModelStorageStrategyConfig(name="PyTorchFullModel")
        full_model_strategy.zip = True
        full_model_strategy.zip_algorithm = "ZIP_DEFLATED"
        inc_model_strategy = ModelStorageStrategyConfig(name="WeightsDifference")
        inc_model_strategy.zip = False
        inc_model_strategy.config = json.dumps({"operator": "sub"})
        complex_pipeline = database.register_pipeline(
            75, "ResNet18", json.dumps({"num_classes": 10}), True, full_model_strategy, inc_model_strategy, 10
        )

    manager = ModelStorageManager(get_modyn_config(), pathlib.Path("storage"))

    strategy = manager.get_model_storage_strategy(simple_pipeline)
    assert strategy.incremental_model_strategy is None
    assert strategy.full_model_interval is None
    assert not strategy.full_model_strategy.zip

    complex_strategy = manager.get_model_storage_strategy(complex_pipeline)
    assert complex_strategy.full_model_strategy.zip
    assert complex_strategy.full_model_strategy.zip_algorithm == ZIP_DEFLATED
    assert complex_strategy.incremental_model_strategy
    assert not complex_strategy.incremental_model_strategy.zip


@patch("modyn.model_storage.internal.model_storage_manager.current_time_millis", return_value=100)
@patch.object(ModelStorageManager, "_get_base_model_state", return_value=MockModel().state_dict())
def test_store_model(base_model_mock, current_time_mock):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_directory_path = pathlib.Path(temp_dir)
        manager = ModelStorageManager(get_modyn_config(), temp_directory_path)

        with MetadataDatabaseConnection(get_modyn_config()) as database:
            parent_id = database.add_trained_model(1, 128, "before.model")

        model_storage_strategy = manager.get_model_storage_strategy(1)
        model_storage_strategy.full_model_strategy.save_model(
            MockModel().state_dict(), temp_directory_path / "before.model"
        )

        torch.save(
            {"model": get_mock_model_after().state_dict(), "metadata": True}, temp_directory_path / "model.modyn"
        )

        model_id = manager.store_model(1, 129, temp_directory_path / "model.modyn")

        with MetadataDatabaseConnection(get_modyn_config()) as database:
            model: TrainedModel = database.session.get(TrainedModel, model_id)

            assert model.pipeline_id == 1
            assert model.trigger_id == 129
            assert model.model_path == "100_1_129.model"
            assert model.parent_model == parent_id
            assert model.metadata_path == "100_1_129.metadata.zip"

        with open(temp_directory_path / model.model_path, "rb") as model_file:
            assert model_file.read() == b"\x00\x00\x00\x40"

        unzip_file(temp_directory_path / model.metadata_path, temp_directory_path / "unzipped.metadata")
        assert torch.load(temp_directory_path / "unzipped.metadata")["metadata"]

        loaded_model = manager.load_model(model_id, True)

        assert loaded_model["model"]["_weight"].item() == 3
        assert loaded_model["metadata"]


def test_store_model_resnet():
    full_model_strategy = ModelStorageStrategyConfig(name="CompressedFullModel")
    full_model_strategy.zip = True

    with MetadataDatabaseConnection(get_modyn_config()) as database:
        pipeline_id = database.register_pipeline(
            1, "ResNet18", json.dumps({"num_classes": 10}), True, full_model_strategy
        )

    resnet = ResNet18(model_configuration={"num_classes": 10}, device="cpu", amp=False)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_directory_path = pathlib.Path(temp_dir)
        manager = ModelStorageManager(get_modyn_config(), temp_directory_path)

        torch.save({"model": resnet.model.state_dict(), "metadata": True}, temp_directory_path / "model.modyn")

        model_id = manager.store_model(pipeline_id, 1, temp_directory_path / "model.modyn")
        loaded_state = manager.load_model(model_id, True)
        assert loaded_state["metadata"]

        original_state = resnet.model.state_dict()
        for layer_name, _ in loaded_state["model"].items():
            assert torch.all(torch.eq(loaded_state["model"][layer_name], original_state[layer_name]))


@patch.object(ModelStorageManager, "_get_base_model_state", return_value=MockModel().state_dict())
def test_load_model(base_model_mock: MagicMock):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_directory_path = pathlib.Path(temp_dir)
        manager = ModelStorageManager(get_modyn_config(), temp_directory_path)

        model_file_name = "mock.model"
        with MetadataDatabaseConnection(get_modyn_config()) as database:
            model_id = database.add_trained_model(1, 32, model_file_name)

        model_storage_strategy = manager.get_model_storage_strategy(1)
        model_storage_strategy.full_model_strategy.save_model(
            get_mock_model_after().state_dict(), temp_directory_path / model_file_name
        )

        reconstructed_state = manager.load_model(model_id, False)

        assert reconstructed_state["model"]["_weight"].item() == 3


@patch.object(ModelStorageManager, "_get_base_model_state", return_value=MockModel().state_dict())
def test_load_model_metadata(base_model_mock: MagicMock):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_directory_path = pathlib.Path(temp_dir)
        manager = ModelStorageManager(get_modyn_config(), temp_directory_path)

        model_file_name = "mock.model"
        with MetadataDatabaseConnection(get_modyn_config()) as database:
            model_id = database.add_trained_model(1, 32, model_file_name, "mock.metadata.zip")

        model_storage_strategy = manager.get_model_storage_strategy(1)
        model_storage_strategy.full_model_strategy.save_model(
            get_mock_model_after().state_dict(), temp_directory_path / model_file_name
        )
        torch.save({"metadata": True}, temp_directory_path / "mock.metadata")
        zip_file(temp_directory_path / "mock.metadata", temp_directory_path / "mock.metadata.zip")

        reconstructed_state = manager.load_model(model_id, True)

        assert reconstructed_state["model"]["_weight"].item() == 3
        assert reconstructed_state["metadata"]


@patch.object(ModelStorageManager, "_get_base_model_state", return_value=MockModel().state_dict())
def test_load_model_invalid(base_model_mock: MagicMock):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_directory_path = pathlib.Path(temp_dir)
        manager = ModelStorageManager(get_modyn_config(), temp_directory_path)

        assert manager.load_model(133, False) is None

        model_file_name = "mock.model"
        with MetadataDatabaseConnection(get_modyn_config()) as database:
            model_id = database.add_trained_model(1, 23, model_file_name)

        model_storage_strategy = manager.get_model_storage_strategy(1)
        model_storage_strategy.full_model_strategy.save_model(
            get_mock_model_after().state_dict(), temp_directory_path / model_file_name
        )

        assert manager.load_model(model_id, True) is None


@patch.object(ModelStorageManager, "_get_base_model_state", return_value=MockModel().state_dict())
def test_delete_model(base_model_mock: MagicMock):
    mock_model = MockModel()
    model_state = mock_model.state_dict()
    model_state_after = get_mock_model_after().state_dict()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_directory_path = pathlib.Path(temp_dir)

        with MetadataDatabaseConnection(get_modyn_config()) as database:
            parent_id = database.add_trained_model(1, 52, "parent.modyn")
            first_child_id = database.add_trained_model(1, 53, "child1.modyn", parent_model=parent_id)
            second_child_id = database.add_trained_model(1, 54, "child2.modyn", parent_model=parent_id)

        manager = ModelStorageManager(get_modyn_config(), temp_directory_path)
        model_storage_strategy = manager.get_model_storage_strategy(1)
        model_storage_strategy.full_model_strategy.save_model(model_state, temp_directory_path / "parent.modyn")
        model_storage_strategy.incremental_model_strategy.save_model(
            model_state_after, model_state, temp_directory_path / "child1.modyn"
        )
        model_storage_strategy.incremental_model_strategy.save_model(
            model_state_after, model_state, temp_directory_path / "child2.modyn"
        )

        success = manager.delete_model(parent_id)

        assert success
        assert not (temp_directory_path / "parent.modyn").exists()

        with MetadataDatabaseConnection(get_modyn_config()) as database:
            first_child: TrainedModel = database.session.get(TrainedModel, first_child_id)
            second_child: TrainedModel = database.session.get(TrainedModel, second_child_id)

            assert first_child.parent_model is None
            assert second_child.parent_model is None

        assert manager.load_model(first_child_id, False)["model"]["_weight"] == 3
        assert manager.load_model(second_child_id, False)["model"]["_weight"] == 3
        assert not manager.delete_model(-1)
