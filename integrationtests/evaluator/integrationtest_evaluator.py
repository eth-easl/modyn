import json
import pathlib
import time

import torch
from integrationtests.utils import ImageDatasetHelper, connect_to_server, get_modyn_config
from modyn.common.ftp import upload_file
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    DatasetInfo,
    EvaluateModelRequest,
    EvaluateModelResponse,
    EvaluationAbortedReason,
    EvaluationResultRequest,
    EvaluationResultResponse,
    EvaluationStatusRequest,
    EvaluationStatusResponse,
)
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import JsonString as EvaluatorJsonString
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import MetricConfiguration
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import PythonString as EvaluatorPythonString
from modyn.evaluator.internal.grpc.generated.evaluator_pb2_grpc import EvaluatorStub
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import Trigger
from modyn.metadata_database.utils import ModelStorageStrategyConfig
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import RegisterModelRequest, RegisterModelResponse
from modyn.model_storage.internal.grpc.generated.model_storage_pb2_grpc import ModelStorageStub
from modyn.models import ResNet18
from modyn.utils import calculate_checksum

TEST_MODELS_PATH = pathlib.Path("/app") / "model_storage" / "test_models"
MODYN_CONFIG = get_modyn_config()
DATASET_ID = "image_test_dataset"

POLL_INTERVAL = 1
POLL_TIMEOUT = 60


def wait_for_evaluation(evaluation_id: int, evaluator: EvaluatorStub) -> EvaluationResultResponse:
    start_time = time.time()
    is_running = True
    while time.time() - start_time < POLL_TIMEOUT:
        req = EvaluationStatusRequest(evaluation_id=evaluation_id)
        response: EvaluationStatusResponse = evaluator.get_evaluation_status(req)
        is_running = response.is_running
        if not is_running:
            break
        time.sleep(POLL_INTERVAL)

    if is_running:
        raise TimeoutError("Evaluation did not finish in time")
    req = EvaluationResultRequest(evaluation_id=evaluation_id)
    return evaluator.get_evaluation_result(req)


def prepare_dataset(dataset_helper: ImageDatasetHelper):
    dataset_helper.add_images_to_dataset(start_number=0, end_number=5)
    split_ts1 = int(time.time()) + 1
    # we want to make sure the split_ts properly splits the dataset
    time.sleep(2)
    dataset_helper.add_images_to_dataset(start_number=5, end_number=12)
    split_ts2 = int(time.time()) + 1
    time.sleep(2)
    dataset_helper.add_images_to_dataset(start_number=12, end_number=22)
    return split_ts1, split_ts2, 5, 7, 10


def prepare_model():
    model_file_name = "test_model.modyn"
    pathlib.Path(TEST_MODELS_PATH).mkdir(parents=True, exist_ok=True)
    model = ResNet18(model_configuration={"num_classes": 10}, device="cpu", amp=False)
    torch.save({"model": model.model.state_dict(), "metadata": True}, TEST_MODELS_PATH / model_file_name)
    upload_file(
        MODYN_CONFIG["trainer_server"]["hostname"],
        int(MODYN_CONFIG["trainer_server"]["ftp_port"]),
        "modyn",
        "modyn",
        local_file_path=TEST_MODELS_PATH / model_file_name,
        remote_file_path=pathlib.Path(model_file_name),
    )

    # add corresponding trigger and pipeline to database
    with MetadataDatabaseConnection(MODYN_CONFIG) as database:
        pipeline_id = database.register_pipeline(
            2,
            "ResNet18",
            json.dumps({"num_classes": 10}),
            False,
            "{}",
            ModelStorageStrategyConfig("PyTorchFullModel"),
            None,
            None,
        )

        trigger = Trigger(trigger_id=0, pipeline_id=pipeline_id)
        database.session.add(trigger)
        database.session.commit()

    model_storage_channel = connect_to_server("model_storage")
    model_storage = ModelStorageStub(model_storage_channel)
    register_model_request = RegisterModelRequest(
        pipeline_id=pipeline_id,
        trigger_id=trigger.trigger_id,
        hostname=MODYN_CONFIG["trainer_server"]["hostname"],
        port=int(MODYN_CONFIG["trainer_server"]["ftp_port"]),
        model_path=model_file_name,
        checksum=calculate_checksum(TEST_MODELS_PATH / model_file_name),
    )
    register_response: RegisterModelResponse = model_storage.RegisterModel(register_model_request)
    assert register_response.success, "Could not register the model"
    return register_response.model_id, pipeline_id, trigger.trigger_id


def evaluate_model(
    model_id: int, start_timestamp: int, end_timestamp: int, evaluator: EvaluatorStub
) -> EvaluateModelResponse:
    eval_transform_function = "import torch\n"
    "def evaluation_transformer_function(model_output: torch.Tensor) -> torch.Tensor:\n\t"
    "return torch.argmax(model_output, dim=-1)"

    bytes_parser = "from PIL import Image\nimport io\n"
    "def bytes_parser_function(data: memoryview) -> Image:\n\t"
    "return Image.open(io.BytesIO(data)).convert('RGB')"

    request = EvaluateModelRequest(
        model_id=model_id,
        dataset_info=DatasetInfo(
            dataset_id=DATASET_ID,
            num_dataloaders=1,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        ),
        device="cpu",
        batch_size=2,
        metrics=[
            MetricConfiguration(
                name="Accuracy",
                config=EvaluatorJsonString(value="{}"),
                evaluation_transformer=EvaluatorPythonString(value=eval_transform_function),
            )
        ],
        transform_list=["transforms.ToTensor()", "transforms.Normalize((0.1307,), (0.3081,))"],
        bytes_parser=EvaluatorPythonString(value=bytes_parser),
        label_transformer=EvaluatorPythonString(value=""),
    )

    return evaluator.evaluate_model(request)


def test_evaluator(dataset_helper: ImageDatasetHelper):
    evaluator_channel = connect_to_server("evaluator")
    evaluator = EvaluatorStub(evaluator_channel)
    split_ts1, split_ts2, split1_size, split2_size, split3_size = prepare_dataset(dataset_helper)
    model_id, pipeline_id, trigger_id = prepare_model()

    response = evaluate_model(model_id, split_ts2, split_ts1, evaluator)
    assert not response.evaluation_started, "Evaluation should not start if start_timestamp > end_timestamp"
    assert response.dataset_size == 0
    assert response.eval_aborted_reason == EvaluationAbortedReason.EMPTY_DATASET

    response = evaluate_model(model_id, split_ts1, split_ts2, evaluator)
    assert response.evaluation_started
    assert response.dataset_size == split2_size

    wait_for_evaluation(response.evaluation_id, evaluator)


if __name__ == "__main__":
    dataset_helper = ImageDatasetHelper(dataset_size=0, dataset_id=DATASET_ID)
    try:
        dataset_helper.setup_dataset()
        test_evaluator(dataset_helper)
    finally:
        dataset_helper.cleanup_dataset_dir()
        dataset_helper.cleanup_storage_database()
