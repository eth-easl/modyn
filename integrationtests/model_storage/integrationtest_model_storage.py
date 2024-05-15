# end-to-end testing of the model storage component
import io
import json
import logging
import pathlib
import shutil
from typing import Optional, Tuple

import grpc
import torch
from integrationtests.utils import get_modyn_config
from modyn.common.ftp import delete_file, download_trained_model, upload_file
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import Pipeline, Trigger
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

FILE_NAME_PARENT = "test_parent.modyn"
MODEL_PARENT = ResNet18(model_configuration={"num_classes": 10}, device="cpu", amp=False)

FILE_NAME_CHILD = "test_child.modyn"
MODEL_CHILD = ResNet18(model_configuration={"num_classes": 10}, device="cpu", amp=False)


def create_dummy_file():
    pathlib.Path(TEST_MODELS_PATH).mkdir(parents=True, exist_ok=True)

    for model, file_name in [(MODEL_PARENT, FILE_NAME_PARENT), (MODEL_CHILD, FILE_NAME_CHILD)]:
        torch.save({"model": model.model.state_dict(), "metadata": True}, TEST_MODELS_PATH / file_name)


def cleanup_models_dir() -> None:
    shutil.rmtree(TEST_MODELS_PATH)


def connect_to_model_storage(config: dict) -> grpc.Channel:
    model_storage_address = f"{config['model_storage']['hostname']}:{config['model_storage']['port']}"
    model_storage_channel = grpc.insecure_channel(model_storage_address)

    if not grpc_connection_established(model_storage_channel) or model_storage_channel is None:
        raise ConnectionError(f"Could not establish gRPC connection to component at {model_storage_address}.")

    return model_storage_channel


def upload_dummy_files_to_trainer(config: dict):
    for file_name in [FILE_NAME_PARENT, FILE_NAME_CHILD]:
        upload_file(
            config["trainer_server"]["hostname"],
            int(config["trainer_server"]["ftp_port"]),
            "modyn",
            "modyn",
            local_file_path=TEST_MODELS_PATH / file_name,
            remote_file_path=pathlib.Path(file_name),
        )


def delete_dummy_files_from_trainer(config: dict):
    for file_name in [FILE_NAME_PARENT, FILE_NAME_CHILD]:
        delete_file(
            config["trainer_server"]["hostname"],
            int(config["trainer_server"]["ftp_port"]),
            "modyn",
            "modyn",
            pathlib.Path(file_name),
        )


def insert_triggers_into_database(
    modyn_config: dict,
    full_strategy: ModelStorageStrategyConfig,
    inc_strategy: Optional[ModelStorageStrategyConfig],
    full_model_interval: Optional[int],
) -> Tuple[int, int, int]:
    parrent_trigger_id = 0
    child_trigger_id = 1
    with MetadataDatabaseConnection(modyn_config) as database:
        pipeline_id = database.register_pipeline(
            2,
            "ResNet18",
            json.dumps({"num_classes": 10}),
            False,
            "{}",
            full_strategy,
            inc_strategy,
            full_model_interval,
        )

        trigger_parent = Trigger(trigger_id=parrent_trigger_id, pipeline_id=pipeline_id)
        trigger_child = Trigger(trigger_id=child_trigger_id, pipeline_id=pipeline_id)
        database.session.add(trigger_parent)
        database.session.add(trigger_child)
        database.session.commit()

    return pipeline_id, parrent_trigger_id, child_trigger_id


def delete_data_from_database(modyn_config: dict, pipeline_id: int):
    with MetadataDatabaseConnection(modyn_config) as database:
        database.session.query(Trigger).filter(
            Trigger.pipeline_id == pipeline_id,
        ).delete()
        database.session.query(Pipeline).filter(Pipeline.pipeline_id == pipeline_id).delete()
        database.session.commit()


def check_loaded_model(path: pathlib.Path, original_model_state: dict) -> None:
    with open(path, "rb") as state_file:
        checkpoint = torch.load(io.BytesIO(state_file.read()))

    assert "model" in checkpoint, "Model state is not stored in file"
    resnet = ResNet18(model_configuration={"num_classes": 10}, device="cpu", amp=False)
    resnet.model.load_state_dict(checkpoint["model"])

    assert checkpoint["metadata"]

    loaded_state = resnet.model.state_dict()
    for layer_name, _ in resnet.model.state_dict().items():
        assert torch.allclose(loaded_state[layer_name], original_model_state[layer_name], rtol=1e-04, atol=1e-05)


def download_and_check_model(
    pipeline_id: int,
    trigger_id: int,
    modyn_config: dict,
    model_storage: ModelStorageStub,
    file_name: str,
    original_model_state: dict,
) -> int:
    # try to register a new model at model storage
    request_register = RegisterModelRequest(
        pipeline_id=pipeline_id,
        trigger_id=trigger_id,
        hostname=modyn_config["trainer_server"]["hostname"],
        port=int(modyn_config["trainer_server"]["ftp_port"]),
        model_path=file_name,
        checksum=calculate_checksum(TEST_MODELS_PATH / file_name),
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
        modyn_config["model_storage"],
        remote_path=pathlib.Path(response_fetch.model_path),
        checksum=response_fetch.checksum,
        identifier=42,
        base_directory=TEST_MODELS_PATH,
    )

    assert downloaded_path is not None

    # compare if content matches initial dummy file & delete it
    check_loaded_model(downloaded_path, original_model_state)
    downloaded_path.unlink()

    return model_id


def test_model_storage(
    modyn_config: dict,
    full_strategy: ModelStorageStrategyConfig,
    inc_strategy: Optional[ModelStorageStrategyConfig],
    full_model_interval: Optional[int],
):
    # register pipeline and trigger
    pipeline_id, parent_trigger, child_trigger = insert_triggers_into_database(
        modyn_config, full_strategy, inc_strategy, full_model_interval
    )

    with MetadataDatabaseConnection(modyn_config) as database:
        model_class_name, model_config, amp = database.get_model_configuration(pipeline_id)

    assert model_class_name == "ResNet18"
    assert json.loads(model_config) == {"num_classes": 10}
    assert not amp

    model_storage_channel = connect_to_model_storage(modyn_config)
    model_storage = ModelStorageStub(model_storage_channel)

    parent_id = download_and_check_model(
        pipeline_id, parent_trigger, modyn_config, model_storage, FILE_NAME_PARENT, MODEL_PARENT.model.state_dict()
    )

    child_id = download_and_check_model(
        pipeline_id, child_trigger, modyn_config, model_storage, FILE_NAME_CHILD, MODEL_CHILD.model.state_dict()
    )

    if inc_strategy is not None:
        # try to delete parent on model storage
        request_delete = DeleteModelRequest(model_id=parent_id)
        response_delete: DeleteModelResponse = model_storage.DeleteModel(request_delete)

        assert not response_delete.success

    # delete child on model storage
    request_delete = DeleteModelRequest(model_id=child_id)
    response_delete: DeleteModelResponse = model_storage.DeleteModel(request_delete)

    assert response_delete.success

    # fetch a (now) invalid model
    request_invalid_fetch = FetchModelRequest(model_id=child_id)
    response_invalid_fetch: FetchModelResponse = model_storage.FetchModel(request_invalid_fetch)

    assert not response_invalid_fetch.success

    # delete a (now) invalid model
    request_invalid_delete = DeleteModelRequest(model_id=child_id)
    response_invalid_delete: DeleteModelResponse = model_storage.DeleteModel(request_invalid_delete)

    assert not response_invalid_delete.success

    # delete parent on model storage
    request_delete = DeleteModelRequest(model_id=parent_id)
    response_delete: DeleteModelResponse = model_storage.DeleteModel(request_delete)

    assert response_delete.success

    # clean-up database
    delete_data_from_database(modyn_config, pipeline_id)


def main() -> None:
    modyn_config = get_modyn_config()

    pytorch_full = ModelStorageStrategyConfig("PyTorchFullModel")

    compressed_full = ModelStorageStrategyConfig("BinaryFullModel")
    compressed_full.zip = True
    compressed_full.zip_algorithm = "ZIP_LZMA"

    sub_delta_inc = ModelStorageStrategyConfig("WeightsDifference")
    sub_delta_inc.config = json.dumps({"operator": "sub"})

    xor_full = ModelStorageStrategyConfig("WeightsDifference")
    xor_full.zip = True
    xor_full.config = json.dumps({"operator": "xor", "split_exponent": True, "rle": True})

    policies = [
        (pytorch_full, None, None),
        (compressed_full, sub_delta_inc, 5),
        (pytorch_full, xor_full, 5),
    ]
    try:
        create_dummy_file()
        upload_dummy_files_to_trainer(modyn_config)

        for policy in policies:
            test_model_storage(modyn_config, *policy)
    finally:
        delete_dummy_files_from_trainer(modyn_config)
        cleanup_models_dir()


if __name__ == "__main__":
    main()
