# end-to-end testing of the model storage component
import pathlib
import shutil

import grpc
from integrationtests.utils import get_modyn_config
from modyn.common.ftp import delete_file, download_file, upload_file
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import TrainedModel, Trigger
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import (
    DeleteModelRequest,
    DeleteModelResponse,
    FetchModelRequest,
    FetchModelResponse,
    RegisterModelRequest,
    RegisterModelResponse,
)
from modyn.model_storage.internal.grpc.generated.model_storage_pb2_grpc import ModelStorageStub
from modyn.utils import grpc_connection_established

TEST_MODELS_PATH = pathlib.Path("/app") / "model_storage" / "test_models"
TEST_FILE_NAME_LOCAL = "test_model_local.txt"
TEST_FILE_NAME_LOCAL_RESP = "test_model_local_response.txt"
TEST_FILE_NAME_REMOTE = "test_model_remote.txt"


def create_dummy_file():
    pathlib.Path(TEST_MODELS_PATH).mkdir(parents=True, exist_ok=True)

    with open(TEST_MODELS_PATH / TEST_FILE_NAME_LOCAL, "w") as f:
        f.write("Test model storage component")


def cleanup_models_dir() -> None:
    shutil.rmtree(TEST_MODELS_PATH)


def connect_to_model_storage(config: dict) -> grpc.Channel:
    model_storage_address = f"{config['model_storage']['hostname']}:{config['model_storage']['port']}"
    model_storage_channel = grpc.insecure_channel(model_storage_address)

    if not grpc_connection_established(model_storage_channel) or model_storage_channel is None:
        raise ConnectionError(f"Could not establish gRPC connection to component at {model_storage_address}.")

    return model_storage_channel


def upload_dummy_file_to_trainer(config: dict):
    upload_file(
        config["trainer_server"]["hostname"],
        int(config["trainer_server"]["ftp_port"]),
        "modyn",
        "modyn",
        local_file_path=TEST_MODELS_PATH / TEST_FILE_NAME_LOCAL,
        remote_file_path=pathlib.Path(TEST_FILE_NAME_REMOTE),
    )


def delete_dummy_file_from_trainer(config: dict):
    delete_file(
        config["trainer_server"]["hostname"],
        int(config["trainer_server"]["ftp_port"]),
        "modyn",
        "modyn",
        pathlib.Path(TEST_FILE_NAME_REMOTE),
    )


def insert_trigger_into_database(config: dict) -> (int, int):
    with MetadataDatabaseConnection(config) as database:
        pipeline_id = database.register_pipeline(2)

        trigger = Trigger(trigger_id=10, pipeline_id=pipeline_id)
        database.session.add(trigger)
        database.session.commit()

        return trigger.pipeline_id, trigger.trigger_id


def delete_data_from_database(config: dict, pipeline_id: int, trigger_id: int, model_id: int):
    with MetadataDatabaseConnection(config) as database:
        database.session.query(TrainedModel).filter(TrainedModel.model_id == model_id).delete()
        database.session.query(Trigger).filter(
            Trigger.pipeline_id == pipeline_id and Trigger.trigger_id == trigger_id
        ).delete()
        database.session.commit()


def test_model_storage(config: dict):
    # register pipeline and trigger
    pipeline_id, trigger_id = insert_trigger_into_database(config)

    model_storage_channel = connect_to_model_storage(config)
    model_storage = ModelStorageStub(model_storage_channel)

    # try to register a new model in the model storage
    request_register = RegisterModelRequest(
        pipeline_id=pipeline_id,
        trigger_id=trigger_id,
        hostname=config["trainer_server"]["hostname"],
        port=int(config["trainer_server"]["ftp_port"]),
        model_path=str(TEST_FILE_NAME_REMOTE),
    )
    response_register: RegisterModelResponse = model_storage.RegisterModel(request_register)

    assert response_register.success, "Could not register a new model"
    model_id = response_register.model_id

    # try to fetch the registered model
    request_fetch = FetchModelRequest(model_id=model_id)
    response_fetch: FetchModelResponse = model_storage.FetchModel(request_fetch)
    model_path = pathlib.Path(response_fetch.model_path)

    assert response_fetch.success, "Could not find model with this id"

    # download the model (dummy file) from model storage
    download_file(
        config["model_storage"]["hostname"],
        int(config["model_storage"]["ftp_port"]),
        "modyn",
        "modyn",
        remote_file_path=model_path,
        local_file_path=TEST_MODELS_PATH / TEST_FILE_NAME_LOCAL_RESP,
    )

    # compare if content matches initial dummy file
    with open(TEST_MODELS_PATH / TEST_FILE_NAME_LOCAL_RESP, "r") as resp_file:
        assert resp_file.read() == "Test model storage component", "File contents do not match"

    # delete model on model storage component
    request_delete = DeleteModelRequest(model_id=model_id)
    response_delete: DeleteModelResponse = model_storage.DeleteModel(request_delete)

    assert response_delete.valid

    # fetch a (now) invalid model
    request_invalid_fetch = FetchModelRequest(model_id=model_id)
    response_invalid_fetch: FetchModelResponse = model_storage.FetchModel(request_invalid_fetch)

    assert not response_invalid_fetch.success

    # delete a (now) invalid model
    request_invalid_delete = DeleteModelRequest(model_id=model_id)
    response_invalid_delete: DeleteModelResponse = model_storage.DeleteModel(request_invalid_delete)

    assert not response_invalid_delete.valid

    # clean-up database
    delete_data_from_database(config, pipeline_id, trigger_id, model_id)


def main() -> None:
    modyn_config = get_modyn_config()
    try:
        create_dummy_file()
        upload_dummy_file_to_trainer(modyn_config)
        test_model_storage(modyn_config)
    finally:
        delete_dummy_file_from_trainer(modyn_config)
        cleanup_models_dir()


if __name__ == "__main__":
    main()
