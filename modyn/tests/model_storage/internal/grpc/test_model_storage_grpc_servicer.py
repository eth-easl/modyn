import os
import pathlib
import shutil
import tempfile
from unittest.mock import MagicMock, patch

from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import TrainedModel, Trigger

# pylint: disable-next=no-name-in-module
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import (
    DeleteModelRequest,
    DeleteModelResponse,
    FetchModelRequest,
    FetchModelResponse,
    RegisterModelRequest,
    RegisterModelResponse,
)
from modyn.model_storage.internal.grpc.model_storage_grpc_servicer import ModelStorageGRPCServicer

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

        pipeline_id = database.register_pipeline(1)
        trigger = Trigger(trigger_id=10, pipeline_id=pipeline_id)

        database.session.add(trigger)
        database.session.commit()

        pipeline2 = database.register_pipeline(4)
        trigger2 = Trigger(trigger_id=50, pipeline_id=pipeline2)

        database.session.add(trigger2)
        database.session.commit()


def teardown():
    os.remove(DATABASE)


@patch("modyn.model_storage.internal.grpc.model_storage_grpc_servicer.download_file")
@patch("modyn.model_storage.internal.grpc.model_storage_grpc_servicer.current_time_millis", return_value=100)
def test_register_model(current_time_millis, download_file_mock: MagicMock):  # pylint: disable=unused-argument
    config = get_modyn_config()
    with tempfile.TemporaryDirectory() as storage_dir:
        storage_path = pathlib.Path(storage_dir)

        with open(storage_path / "test.txt", "wb") as file:
            file.write(b"Our test model")

        servicer = ModelStorageGRPCServicer(config, storage_path)
        assert servicer is not None

        req = RegisterModelRequest(
            pipeline_id=1,
            trigger_id=10,
            hostname=config["trainer_server"]["hostname"],
            port=int(config["trainer_server"]["ftp_port"]),
            model_path="test.txt",
        )

        resp: RegisterModelResponse = servicer.RegisterModel(req, None)

        download_file_mock.assert_called_once()
        kwargs = download_file_mock.call_args.kwargs
        remote_file_path = kwargs["remote_file_path"]
        local_file_path = kwargs["local_file_path"]

        shutil.copyfile(storage_path / remote_file_path, local_file_path)

        assert resp.success

        # download file under path {current_time_millis}_{pipeline_id}_{trigger_id}.modyn
        with open(storage_path / f"100_{resp.model_id}_10.modyn", "rb") as file:
            assert file.read().decode("utf-8") == "Our test model"


def test_fetch_model():
    config = get_modyn_config()
    with tempfile.TemporaryDirectory() as storage_dir:
        storage_path = pathlib.Path(storage_dir)

        servicer = ModelStorageGRPCServicer(config, storage_path)
        assert servicer is not None

        with MetadataDatabaseConnection(config) as database:
            model_id = database.add_trained_model(2, 50, "test_model.modyn")

        req = FetchModelRequest(model_id=model_id)
        resp: FetchModelResponse = servicer.FetchModel(req, None)

        assert resp.success
        assert resp.model_path == "test_model.modyn"

        req_invalid = FetchModelRequest(model_id=142)
        resp_invalid: FetchModelResponse = servicer.FetchModel(req_invalid, None)

        assert not resp_invalid.success


def test_delete_model():
    config = get_modyn_config()
    with tempfile.TemporaryDirectory() as storage_dir:
        storage_path = pathlib.Path(storage_dir)

        servicer = ModelStorageGRPCServicer(config, storage_path)
        assert servicer is not None

        with open(storage_path / "model_to_be_deleted.modyn", "wb") as file:
            file.write(b"model that will be deleted")

        assert os.path.isfile(storage_path / "model_to_be_deleted.modyn")

        with MetadataDatabaseConnection(config) as database:
            model_id = database.add_trained_model(2, 50, "model_to_be_deleted.modyn")

        req = DeleteModelRequest(model_id=model_id)
        resp: DeleteModelResponse = servicer.DeleteModel(req, None)

        assert resp.success
        assert not os.path.isfile(storage_path / "model_to_be_deleted.modyn")

        req_invalid = DeleteModelRequest(model_id=model_id)
        resp_invalid: DeleteModelResponse = servicer.DeleteModel(req_invalid, None)

        assert not resp_invalid.success

        with MetadataDatabaseConnection(config) as database:
            model_id = database.session.get(TrainedModel, model_id)

            assert not model_id
