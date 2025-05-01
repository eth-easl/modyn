# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
# ruff: noqa: N802  # grpc functions are not snake case


import platform
from unittest.mock import patch

import grpc
import pytest
import torch
from torchvision import transforms

from modyn.evaluator.internal.dataset.evaluation_dataset import EvaluationDataset
from modyn.models.tokenizers import DistilBertTokenizerTransform
from modyn.storage.internal.grpc.generated.storage_pb2 import (
    GetDataPerWorkerRequest,
    GetDataPerWorkerResponse,
    GetRequest,
    GetResponse,
)


def get_identity_bytes_parser():
    return "def bytes_parser_function(x):\n\treturn x"


def bytes_parser_function(data):
    return data


class MockStorageStub:
    def __init__(self, channel) -> None:
        pass

    def Get(self, request: GetRequest):  # pylint: disable=invalid-name
        for key in request.keys:
            yield GetResponse(
                samples=[key.to_bytes(2, "big"), (key + 1).to_bytes(2, "big")],
                keys=[key, key + 1],
                labels=[5, 5],
                target=[key.to_bytes(2, "big"), (key + 1).to_bytes(2, "big")],
            )

    def GetDataPerWorker(self, request: GetDataPerWorkerRequest):  # pylint: disable=invalid-name
        for i in range(0, 8, 4):
            key = 8 * request.worker_id + i
            yield GetDataPerWorkerResponse(keys=[key, key + 2])


def test_invalid_bytes_parser():
    with pytest.raises(AssertionError):
        EvaluationDataset(
            dataset_id="MNIST",
            bytes_parser="",
            serialized_transforms=[],
            storage_address="localhost:1234",
            evaluation_id=10,
        )._init_transforms()

    with pytest.raises(ValueError):
        EvaluationDataset(
            dataset_id="MNIST",
            bytes_parser="bytes_parser_function=1",
            serialized_transforms=[],
            storage_address="localhost:1234",
            evaluation_id=10,
        )._init_transforms()


def test_init():
    evaluation_dataset = EvaluationDataset(
        dataset_id="MNIST",
        bytes_parser=get_identity_bytes_parser(),
        serialized_transforms=[],
        storage_address="localhost:1234",
        evaluation_id=10,
        tokenizer="DistilBertTokenizerTransform",
    )

    assert evaluation_dataset._evaluation_id == 10
    assert evaluation_dataset._dataset_id == "MNIST"
    assert evaluation_dataset._first_call
    assert evaluation_dataset._bytes_parser_function is None
    assert evaluation_dataset._storagestub is None
    assert isinstance(evaluation_dataset._tokenizer, DistilBertTokenizerTransform)


@patch("modyn.evaluator.internal.dataset.evaluation_dataset.StorageStub", MockStorageStub)
@patch("modyn.evaluator.internal.dataset.evaluation_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_grpc(test_insecure_channel, test_grpc_connection_established):
    evaluation_dataset = EvaluationDataset(
        dataset_id="MNIST",
        bytes_parser=get_identity_bytes_parser(),
        serialized_transforms=[],
        storage_address="localhost:1234",
        evaluation_id=10,
    )

    assert evaluation_dataset._storagestub is None
    evaluation_dataset._init_grpc()
    assert isinstance(evaluation_dataset._storagestub, MockStorageStub)


@patch("modyn.evaluator.internal.dataset.evaluation_dataset.StorageStub", MockStorageStub)
@patch("modyn.evaluator.internal.dataset.evaluation_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_get_keys_from_storage(test_insecure_channel, test_grpc_connection_established):
    evaluation_dataset = EvaluationDataset(
        dataset_id="MNIST",
        bytes_parser=get_identity_bytes_parser(),
        serialized_transforms=[],
        storage_address="localhost:1234",
        evaluation_id=10,
    )

    evaluation_dataset._init_grpc()
    for idx, keys in enumerate(evaluation_dataset._get_keys_from_storage(1, 2)):
        base_key = 8 + 4 * idx
        assert keys == [base_key, base_key + 2]


@patch("modyn.evaluator.internal.dataset.evaluation_dataset.StorageStub", MockStorageStub)
@patch("modyn.evaluator.internal.dataset.evaluation_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_get_data_from_storage(test_insecure_channel, test_grpc_connection_established):
    evaluation_dataset = EvaluationDataset(
        dataset_id="MNIST",
        bytes_parser=get_identity_bytes_parser(),
        serialized_transforms=[],
        storage_address="localhost:1234",
        evaluation_id=10,
    )

    evaluation_dataset._init_grpc()
    for idx, elem in enumerate(evaluation_dataset._get_data_from_storage(range(0, 10, 2))):
        for num_elem, data in enumerate(elem):
            key = idx * 2 + num_elem
            assert data == (key, key.to_bytes(2, "big"), 5, key.to_bytes(2, "big"))


def test_init_transforms():
    evaluation_dataset = EvaluationDataset(
        dataset_id="MNIST",
        bytes_parser="def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')",
        serialized_transforms=[],
        storage_address="localhost:1234",
        evaluation_id=10,
    )

    assert evaluation_dataset._bytes_parser_function is None
    assert evaluation_dataset._transform is None

    with patch.object(evaluation_dataset, "_setup_composed_transform") as tv_ds:
        evaluation_dataset._init_transforms()
        assert evaluation_dataset._bytes_parser_function is not None
        assert evaluation_dataset._bytes_parser_function(b"\x03") == 3

        assert evaluation_dataset._transform is not None

        tv_ds.assert_called_once()


@pytest.mark.parametrize(
    "serialized_transforms,transforms_list",
    [
        pytest.param(
            [
                "transforms.RandomResizedCrop(224)",
                "transforms.RandomAffine(degrees=(0, 90))",
                "transforms.ToTensor()",
                "transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
            ],
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomAffine(degrees=(0, 90)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )
    ],
)
def test__setup_composed_transform(serialized_transforms, transforms_list):
    evaluation_dataset = EvaluationDataset(
        dataset_id="MNIST",
        bytes_parser=get_identity_bytes_parser(),
        serialized_transforms=list(serialized_transforms),
        storage_address="localhost:1234",
        evaluation_id=10,
        tokenizer="DistilBertTokenizerTransform",
    )
    evaluation_dataset._bytes_parser_function = bytes_parser_function
    evaluation_dataset._setup_composed_transform()
    assert isinstance(evaluation_dataset._transform.transforms, list)
    assert evaluation_dataset._transform.transforms[0].__name__ == "bytes_parser_function"
    for transform1, transform2 in zip(evaluation_dataset._transform.transforms[1:-1], transforms_list):
        assert transform1.__dict__ == transform2.__dict__
    assert isinstance(evaluation_dataset._transform.transforms[-1], DistilBertTokenizerTransform)


@patch("modyn.evaluator.internal.dataset.evaluation_dataset.StorageStub", MockStorageStub)
@patch("modyn.evaluator.internal.dataset.evaluation_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_dataset_iter(test_insecure_channel, test_grpc_connection_established):
    evaluation_dataset = EvaluationDataset(
        dataset_id="MNIST",
        bytes_parser=get_identity_bytes_parser(),
        serialized_transforms=[],
        storage_address="localhost:1234",
        evaluation_id=10,
    )

    all_data = list(evaluation_dataset)
    assert [x[0] for x in all_data] == list(range(8))
    assert [x[1] for x in all_data] == [x.to_bytes(2, "big") for x in range(8)]
    assert [x[2] for x in all_data] == [5] * 8


@patch("modyn.evaluator.internal.dataset.evaluation_dataset.StorageStub", MockStorageStub)
@patch("modyn.evaluator.internal.dataset.evaluation_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_dataset_iter_with_parsing(test_insecure_channel, test_grpc_connection_established):
    evaluation_dataset = EvaluationDataset(
        dataset_id="MNIST",
        bytes_parser="def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')",
        serialized_transforms=[],
        storage_address="localhost:1234",
        evaluation_id=42,
    )

    dataset_iter = iter(evaluation_dataset)
    all_data = list(dataset_iter)
    print(all_data)
    assert [x[0] for x in all_data] == list(range(8))
    assert [x[1] for x in all_data] == list(range(8))
    assert [x[2] for x in all_data] == [5] * 8


@patch("modyn.evaluator.internal.dataset.evaluation_dataset.StorageStub", MockStorageStub)
@patch("modyn.evaluator.internal.dataset.evaluation_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_dataloader_dataset(test_insecure_channel, test_grpc_connection_established):
    evaluation_dataset = EvaluationDataset(
        dataset_id="MNIST",
        bytes_parser="def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')",
        serialized_transforms=[],
        storage_address="localhost:1234",
        evaluation_id=5,
    )

    dataloader = torch.utils.data.DataLoader(evaluation_dataset, batch_size=4)
    for i, batch in enumerate(dataloader):
        assert len(batch) == 3
        assert i < 2
        assert batch[0].tolist() == [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
        assert torch.equal(batch[1], torch.Tensor([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]))
        assert torch.equal(batch[2], torch.ones(4, dtype=torch.float64) * 5)


@patch("modyn.evaluator.internal.dataset.evaluation_dataset.StorageStub", MockStorageStub)
@patch("modyn.evaluator.internal.dataset.evaluation_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_dataloader_dataset_multi_worker(test_insecure_channel, test_grpc_connection_established):
    if platform.system() == "Darwin":
        # On macOS, spawn is the default, which loses the mocks
        # Hence the test does not work on macOS, only on Linux.
        return

    evaluation_dataset = EvaluationDataset(
        dataset_id="MNIST",
        bytes_parser="def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')",
        serialized_transforms=[],
        storage_address="localhost:1234",
        evaluation_id=20,
    )
    dataloader = torch.utils.data.DataLoader(evaluation_dataset, batch_size=4, num_workers=4)

    data = list(dataloader)
    data.sort(key=lambda batch_data: batch_data[0].min())

    batch_num = -1
    for batch_num, batch in enumerate(data):
        assert len(batch) == 3
        assert batch[0].tolist() == [4 * batch_num, 4 * batch_num + 1, 4 * batch_num + 2, 4 * batch_num + 3]
        assert torch.equal(
            batch[1], torch.Tensor([4 * batch_num, 4 * batch_num + 1, 4 * batch_num + 2, 4 * batch_num + 3])
        )
        assert torch.equal(batch[2], torch.ones(4, dtype=torch.float64) * 5)

    assert batch_num == 7


def get_main_bytes_parser() -> str:
    return """\
def bytes_parser_function(data: bytes):
    return bytes(data) + b"_MAIN"
"""


def get_target_bytes_parser() -> str:
    return """\
def bytes_parser_function(data: bytes):
    return bytes(data) + b"_TARGET"
"""


def get_main_serialized_transforms() -> list[str]:
    return ["transforms.Lambda(lambda x: x + b'_MAIN_TF')"]


def get_target_serialized_transforms() -> list[str]:
    return ["transforms.Lambda(lambda x: x + b'_TARGET_TF')"]


def fake_get_keys_from_storage(self, worker_id, total_workers):
    yield [10]


def fake_get_data_from_storage(self, keys, worker_id):
    yield [(10, b"SAMPLE", b"LABEL", b"TARGET")]


@patch("modyn.evaluator.internal.dataset.evaluation_dataset.StorageStub", MockStorageStub)
@patch("modyn.evaluator.internal.dataset.evaluation_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_evaluation_dataset_different_target_parsers_and_transforms(mock_insecure_channel, mock_grpc_conn):
    dataset = EvaluationDataset(
        dataset_id="dummy_eval",
        bytes_parser=get_main_bytes_parser(),
        serialized_transforms=get_main_serialized_transforms(),
        storage_address="fake-storage:1234",
        evaluation_id=42,
        include_labels=False,  # Use target branch.
        bytes_parser_target=get_target_bytes_parser(),
        serialized_target_transforms=get_target_serialized_transforms(),
        tokenizer=None,
        start_timestamp=None,
        end_timestamp=None,
    )

    dataset._init_grpc = lambda: None
    dataset._get_keys_from_storage = fake_get_keys_from_storage.__get__(dataset)
    dataset._get_data_from_storage = fake_get_data_from_storage.__get__(dataset)

    results = list(dataset)

    key1, sample1, out_target = results[0]
    assert key1 == 10
    assert sample1.endswith(b"SAMPLE_MAIN_MAIN_TF"), f"Unexpected sample: {sample1}"
    assert out_target.endswith(b"TARGET_TARGET_TARGET_TF"), f"Unexpected target: {out_target}"
