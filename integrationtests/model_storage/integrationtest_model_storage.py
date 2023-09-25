# end-to-end testing of the model storage component
import io
import json
import logging
import pathlib
import shutil

import grpc
import torch
from integrationtests.utils import get_modyn_config
from modyn.common.ftp import delete_file, download_trained_model, upload_file
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import Trigger
from modyn.metadata_database.utils import ModelStorageStrategyConfig
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import (
    DeleteModelRequest,
    DeleteModelResponse,
    FetchModelRequest,
    FetchModelResponse,
    RegisterModelRequest,
    RegisterModelResponse,
)
from modyn.model_storage.internal.grpc.generated.model_storage_pb2_grpc import ModelStorageStub
from modyn.models import ResNet18
from modyn.utils import calculate_checksum, grpc_connection_established

TEST_MODELS_PATH = pathlib.Path("/app") / "model_storage" / "test_models"
TEST_FILE_NAME_LOCAL = "test_model_local.modyn"
TEST_FILE_NAME_LOCAL_RESP = "test_model_local_response.modyn"
TEST_FILE_NAME_REMOTE = "test_model_remote.modyn"
SAMPLE_MODEL = ResNet18(model_configuration={"num_classes": 10}, device="cpu", amp=False)


def create_dummy_file():
    pathlib.Path(TEST_MODELS_PATH).mkdir(parents=True, exist_ok=True)

    torch.save({"model": SAMPLE_MODEL.model.state_dict(), "metadata": True}, TEST_MODELS_PATH / TEST_FILE_NAME_LOCAL)


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
    model_storage_strategy = ModelStorageStrategyConfig("CompressedFullModel")
    model_storage_strategy.zip = True

    with MetadataDatabaseConnection(config) as database:
        pipeline_id = database.register_pipeline(
            2, "ResNet18", json.dumps({"num_classes": 10}), False, model_storage_strategy
        )

        trigger = Trigger(trigger_id=10, pipeline_id=pipeline_id)
        database.session.add(trigger)
        database.session.commit()

        return trigger.pipeline_id, trigger.trigger_id


def delete_data_from_database(config: dict, pipeline_id: int, trigger_id: int):
    with MetadataDatabaseConnection(config) as database:
        database.session.query(Trigger).filter(
            Trigger.pipeline_id == pipeline_id and Trigger.trigger_id == trigger_id
        ).delete()
        database.session.commit()


def check_loaded_model(path: pathlib.Path) -> None:
    with open(path, "rb") as state_file:
        checkpoint = torch.load(io.BytesIO(state_file.read()))

    assert "model" in checkpoint, "Model state is not stored in file"
    resnet = ResNet18(model_configuration={"num_classes": 10}, device="cpu", amp=False)
    resnet.model.load_state_dict(checkpoint["model"])

    assert checkpoint["metadata"]

    loaded_state = resnet.model.state_dict()
    original_state = SAMPLE_MODEL.model.state_dict()
    for layer_name, _ in resnet.model.state_dict().items():
        assert torch.all(torch.eq(loaded_state[layer_name], original_state[layer_name]))


def test_model_storage(config: dict):
    # register pipeline and trigger
    pipeline_id, trigger_id = insert_trigger_into_database(config)

    with MetadataDatabaseConnection(config) as database:
        model_id, model_config, amp = database.get_model_configuration(pipeline_id)

    assert model_id == "ResNet18"
    assert json.loads(model_config) == {"num_classes": 10}
    assert not amp

    model_storage_channel = connect_to_model_storage(config)
    model_storage = ModelStorageStub(model_storage_channel)

    # try to register a new model in the model storage
    request_register = RegisterModelRequest(
        pipeline_id=pipeline_id,
        trigger_id=trigger_id,
        hostname=config["trainer_server"]["hostname"],
        port=int(config["trainer_server"]["ftp_port"]),
        model_path=str(TEST_FILE_NAME_REMOTE),
        checksum=calculate_checksum(TEST_MODELS_PATH / TEST_FILE_NAME_LOCAL),
    )
    response_register: RegisterModelResponse = model_storage.RegisterModel(request_register)

    assert response_register.success, "Could not register a new model"
    model_id = response_register.model_id

    # try to fetch the registered model
    request_fetch = FetchModelRequest(model_id=model_id, load_metadata=True)
    response_fetch: FetchModelResponse = model_storage.FetchModel(request_fetch)

    assert response_fetch.success, "Could not find model with this id"

    # download the model (dummy file) from model storage
    downloaded_path = download_trained_model(
        logging.getLogger(__name__),
        config["model_storage"],
        remote_path=pathlib.Path(response_fetch.model_path),
        checksum=response_fetch.checksum,
        identifier=42,
        base_directory=TEST_MODELS_PATH,
    )

    assert downloaded_path is not None

    # compare if content matches initial dummy file
    check_loaded_model(downloaded_path)

    # delete model on model storage component
    request_delete = DeleteModelRequest(model_id=model_id)
    response_delete: DeleteModelResponse = model_storage.DeleteModel(request_delete)

    assert response_delete.success

    # fetch a (now) invalid model
    request_invalid_fetch = FetchModelRequest(model_id=model_id)
    response_invalid_fetch: FetchModelResponse = model_storage.FetchModel(request_invalid_fetch)

    assert not response_invalid_fetch.success

    # delete a (now) invalid model
    request_invalid_delete = DeleteModelRequest(model_id=model_id)
    response_invalid_delete: DeleteModelResponse = model_storage.DeleteModel(request_invalid_delete)

    assert not response_invalid_delete.success

    # clean-up database
    delete_data_from_database(config, pipeline_id, trigger_id)


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
