import json
import pathlib
import shutil
from ftplib import FTP
from typing import Any

import grpc
from integrationtests.utils import get_modyn_config
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import FetchModelRequest, FetchModelResponse
from modyn.model_storage.internal.grpc.generated.model_storage_pb2_grpc import ModelStorageStub
from modyn.selector.internal.grpc.generated.selector_pb2 import DataInformRequest, JsonString, RegisterPipelineRequest
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
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


def connect_to_component(config: dict, component_name: str) -> grpc.Channel:
    component_address = f"{config[component_name]['hostname']}:{config[component_name]['port']}"
    component_channel = grpc.insecure_channel(component_address)

    if not grpc_connection_established(component_channel) or component_channel is None:
        raise ConnectionError(f"Could not establish gRPC connection to component at {component_address}.")

    return component_channel


def test_fetch_model(config: dict):
    ftp = FTP()
    ftp.connect(config["model_storage"]["hostname"], int(config["model_storage"]["ftp_port"]), timeout=3)
    ftp.login("modyn", "modyn")
    ftp.sendcmd("TYPE i")  # Switch to binary mode

    with open(TEST_MODELS_PATH / TEST_FILE_NAME_LOCAL, "rb") as local_file:
        ftp.storbinary(f"STOR {TEST_FILE_NAME_REMOTE}", local_file)

    pipeline_id, trigger_id = insert_trigger_into_database(config)

    with MetadataDatabaseConnection(config) as database:
        model_id = database.add_trained_model(pipeline_id, trigger_id, str(TEST_FILE_NAME_REMOTE))

    model_storage_channel = connect_to_component(config, "model_storage")
    model_storage = ModelStorageStub(model_storage_channel)

    fetch_req = FetchModelRequest(model_id=model_id)
    fetch_resp: FetchModelResponse = model_storage.FetchModel(fetch_req)

    assert fetch_resp.model_path == str(TEST_FILE_NAME_REMOTE)

    with open(TEST_MODELS_PATH / TEST_FILE_NAME_LOCAL_RESP, "wb") as local_file:

        def write_callback(data: Any) -> None:
            local_file.write(data)

        ftp.retrbinary(f"RETR {fetch_resp.model_path}", write_callback)

    with open(TEST_MODELS_PATH / TEST_FILE_NAME_LOCAL_RESP, "r") as resp_file:
        assert resp_file.read() == "Test model storage component", "File contents do not match"

    ftp.delete(TEST_FILE_NAME_REMOTE)
    ftp.close()


def upload_test_model(config: dict):
    ftp = FTP()
    ftp.connect(config["trainer_server"]["hostname"], int(config["trainer_server"]["ftp_port"]), timeout=3)
    ftp.login("modyn", "modyn")
    ftp.sendcmd("TYPE i")  # Switch to binary mode

    with open(TEST_MODELS_PATH / TEST_FILE_NAME_LOCAL, "rb") as local_file:
        ftp.storbinary(f"STOR {TEST_FILE_NAME_REMOTE}", local_file)

    ftp.close()


def delete_test_model(config: dict):
    ftp = FTP()
    ftp.connect(config["trainer_server"]["hostname"], int(config["trainer_server"]["ftp_port"]), timeout=3)
    ftp.login("modyn", "modyn")
    ftp.delete(TEST_FILE_NAME_REMOTE)
    ftp.close()


def insert_trigger_into_database(config: dict) -> (int, int):
    selector_channel = connect_to_component(config, "selector")
    selector = SelectorStub(selector_channel)

    strategy_config = {
        "name": "NewDataStrategy",
        "maximum_keys_in_memory": 50000,
        "config": {"limit": -1, "reset_after_trigger": False},
    }

    pipeline_id = selector.register_pipeline(
        RegisterPipelineRequest(num_workers=2, selection_strategy=JsonString(value=json.dumps(strategy_config)))
    ).pipeline_id

    trigger_id = selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=[],
            timestamps=[],
            labels=[],
        )
    ).trigger_id

    return pipeline_id, trigger_id


# TODO(#167) test model storage end-to-end
"""
def test_model_download(config: dict):
    pipeline_id, trigger_id = insert_trigger_into_database(config)
    upload_test_model(config)

    model_storage_channel = connect_to_component(config, "model_storage")
    model_storage = ModelStorageStub(model_storage_channel)

    request_register = RegisterModelRequest(
        pipeline_id=pipeline_id, trigger_id=trigger_id, model_path=str(TEST_FILE_NAME_REMOTE))
    response_register: RegisterModelResponse = model_storage.RegisterModel(request_register)

    assert response_register.valid, "Could not register a new model"
    model_id = response_register.model_id

    request_model = FetchModelRequest(model_id=model_id)
    response_model: FetchModelResponse = model_storage.FetchModel(request_model)

    assert response_model.valid, "Could not find model with this id"

    ftp = FTP()
    ftp.connect(
        config["model_storage"]["hostname"], int(config["model_storage"]["ftp_port"]), timeout=3
    )

    ftp.login("modyn", "modyn")
    ftp.sendcmd("TYPE i")  # Switch to binary mode

    with open(TEST_MODELS_PATH / TEST_FILE_NAME_LOCAL_RESP, "wb") as local_file:
        def write_callback(data: Any) -> None:
            local_file.write(data)

        ftp.retrbinary(f"RETR {response_model.model_path}", write_callback)


    with open(TEST_MODELS_PATH / TEST_FILE_NAME_LOCAL_RESP, "r") as resp_file:
        assert resp_file.read() == "Test model storage component", "File contents do not match"

    ftp.delete(response_model.model_path)
    ftp.close()
    delete_test_model(config)
"""


def main() -> None:
    modyn_config = get_modyn_config()
    try:
        create_dummy_file()
        test_fetch_model(modyn_config)
    finally:
        cleanup_models_dir()


if __name__ == "__main__":
    main()
