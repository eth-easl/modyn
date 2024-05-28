# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
import math
import platform
from unittest.mock import patch

import grpc
import pytest
import torch
from modyn.models.tokenizers import DistilBertTokenizerTransform
from modyn.storage.internal.grpc.generated.storage_pb2 import GetRequest, GetResponse
from modyn.supervisor.internal.triggers.trigger_datasets import FixedKeysDataset
from torchvision import transforms

DATASET_ID = "MNIST"
TRIGGER_ID = 42
STORAGE_ADDR = "localhost:1234"
KEYS = list(range(10))


def get_identity_bytes_parser():
    return "def bytes_parser_function(x):\n\treturn x"


def bytes_parser_function(data):
    return data


class MockStorageStub:
    def __init__(self, channel) -> None:
        pass

    def Get(self, request: GetRequest):  # pylint: disable=invalid-name  # noqa: N802
        yield GetResponse(
            samples=[key.to_bytes(2, "big") for key in request.keys], keys=request.keys, labels=[1] * len(request.keys)
        )


def test_invalid_bytes_parser():
    with pytest.raises(AssertionError):
        FixedKeysDataset(
            dataset_id=DATASET_ID,
            bytes_parser="",
            serialized_transforms=[],
            storage_address=STORAGE_ADDR,
            trigger_id=TRIGGER_ID,
            keys=KEYS,
        )._init_transforms()

    with pytest.raises(ValueError):
        FixedKeysDataset(
            dataset_id=DATASET_ID,
            bytes_parser="bytes_parser_function=1",
            serialized_transforms=[],
            storage_address=STORAGE_ADDR,
            trigger_id=TRIGGER_ID,
            keys=KEYS,
        )._init_transforms()


def test_init():
    fixed_keys_dataset = FixedKeysDataset(
        dataset_id=DATASET_ID,
        bytes_parser=get_identity_bytes_parser(),
        serialized_transforms=[],
        storage_address=STORAGE_ADDR,
        trigger_id=TRIGGER_ID,
        keys=KEYS,
        tokenizer="DistilBertTokenizerTransform",
    )

    assert fixed_keys_dataset._trigger_id == TRIGGER_ID
    assert fixed_keys_dataset._dataset_id == DATASET_ID
    assert fixed_keys_dataset._first_call
    assert fixed_keys_dataset._bytes_parser_function is None
    assert fixed_keys_dataset._storagestub is None
    assert isinstance(fixed_keys_dataset._tokenizer, DistilBertTokenizerTransform)
    assert fixed_keys_dataset._keys == KEYS


@patch("modyn.supervisor.internal.triggers.trigger_datasets.fixed_keys_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.supervisor.internal.triggers.trigger_datasets.fixed_keys_dataset.grpc_connection_established",
    return_value=True,
)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_grpc(test_insecure_channel, test_grpc_connection_established):
    fixed_keys_dataset = FixedKeysDataset(
        dataset_id=DATASET_ID,
        bytes_parser=get_identity_bytes_parser(),
        serialized_transforms=[],
        storage_address=STORAGE_ADDR,
        trigger_id=TRIGGER_ID,
        keys=KEYS,
    )

    assert fixed_keys_dataset._storagestub is None
    fixed_keys_dataset._init_grpc()
    assert isinstance(fixed_keys_dataset._storagestub, MockStorageStub)


@patch("modyn.supervisor.internal.triggers.trigger_datasets.fixed_keys_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.supervisor.internal.triggers.trigger_datasets.fixed_keys_dataset.grpc_connection_established",
    return_value=True,
)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_get_data_from_storage(test_insecure_channel, test_grpc_connection_established):
    fixed_keys_dataset = FixedKeysDataset(
        dataset_id=DATASET_ID,
        bytes_parser=get_identity_bytes_parser(),
        serialized_transforms=[],
        storage_address=STORAGE_ADDR,
        trigger_id=TRIGGER_ID,
        keys=KEYS,
    )

    fixed_keys_dataset._init_grpc()
    all_data = fixed_keys_dataset._get_data_from_storage(KEYS)
    for data in all_data:
        for i, d in enumerate(data):
            assert d == (i, i.to_bytes(2, "big"), 1)


def test_init_transforms():
    fixed_keys_dataset = FixedKeysDataset(
        dataset_id=DATASET_ID,
        bytes_parser="def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')",
        serialized_transforms=[],
        storage_address=STORAGE_ADDR,
        trigger_id=TRIGGER_ID,
        keys=KEYS,
    )

    assert fixed_keys_dataset._bytes_parser_function is None
    assert fixed_keys_dataset._transform is None

    with patch.object(fixed_keys_dataset, "_setup_composed_transform") as tv_ds:
        fixed_keys_dataset._init_transforms()
        assert fixed_keys_dataset._bytes_parser_function is not None
        assert fixed_keys_dataset._bytes_parser_function(b"\x03") == 3

        assert fixed_keys_dataset._transform is not None

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
    fixed_keys_dataset = FixedKeysDataset(
        dataset_id=DATASET_ID,
        bytes_parser=get_identity_bytes_parser(),
        serialized_transforms=list(serialized_transforms),
        storage_address=STORAGE_ADDR,
        trigger_id=TRIGGER_ID,
        keys=KEYS,
        tokenizer="DistilBertTokenizerTransform",
    )
    fixed_keys_dataset._bytes_parser_function = bytes_parser_function
    fixed_keys_dataset._setup_composed_transform()
    assert isinstance(fixed_keys_dataset._transform.transforms, list)
    assert fixed_keys_dataset._transform.transforms[0].__name__ == "bytes_parser_function"
    for transform1, transform2 in zip(fixed_keys_dataset._transform.transforms[1:-1], transforms_list):
        assert transform1.__dict__ == transform2.__dict__
    assert isinstance(fixed_keys_dataset._transform.transforms[-1], DistilBertTokenizerTransform)


@patch("modyn.supervisor.internal.triggers.trigger_datasets.fixed_keys_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.supervisor.internal.triggers.trigger_datasets.fixed_keys_dataset.grpc_connection_established",
    return_value=True,
)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_dataset_iter(test_insecure_channel, test_grpc_connection_established):
    fixed_keys_dataset = FixedKeysDataset(
        dataset_id=DATASET_ID,
        bytes_parser=get_identity_bytes_parser(),
        serialized_transforms=[],
        storage_address=STORAGE_ADDR,
        trigger_id=TRIGGER_ID,
        keys=KEYS,
    )

    all_data = list(fixed_keys_dataset)
    assert [x[0] for x in all_data] == KEYS
    assert [x[1] for x in all_data] == [x.to_bytes(2, "big") for x in KEYS]
    assert [x[2] for x in all_data] == [1] * len(KEYS)


@patch("modyn.supervisor.internal.triggers.trigger_datasets.fixed_keys_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.supervisor.internal.triggers.trigger_datasets.fixed_keys_dataset.grpc_connection_established",
    return_value=True,
)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_dataset_iter_with_parsing(test_insecure_channel, test_grpc_connection_established):
    fixed_keys_dataset = FixedKeysDataset(
        dataset_id=DATASET_ID,
        bytes_parser="def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')",
        serialized_transforms=[],
        storage_address=STORAGE_ADDR,
        trigger_id=TRIGGER_ID,
        keys=KEYS,
    )
    dataset_iter = iter(fixed_keys_dataset)
    all_data = list(dataset_iter)
    assert [x[0] for x in all_data] == KEYS
    assert [x[1] for x in all_data] == KEYS
    assert [x[2] for x in all_data] == [1] * len(KEYS)


@patch("modyn.supervisor.internal.triggers.trigger_datasets.fixed_keys_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.supervisor.internal.triggers.trigger_datasets.fixed_keys_dataset.grpc_connection_established",
    return_value=True,
)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_dataloader_dataset(test_insecure_channel, test_grpc_connection_established):
    fixed_keys_dataset = FixedKeysDataset(
        dataset_id=DATASET_ID,
        bytes_parser="def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')",
        serialized_transforms=[],
        storage_address=STORAGE_ADDR,
        trigger_id=TRIGGER_ID,
        keys=KEYS,
    )

    dataloader = torch.utils.data.DataLoader(fixed_keys_dataset, batch_size=2)
    for i, batch in enumerate(dataloader):
        assert len(batch) == 3
        assert i < math.floor(len(KEYS) / 2)
        assert batch[0].tolist() == [2 * i, 2 * i + 1]
        assert torch.equal(batch[1], torch.Tensor([2 * i, 2 * i + 1]))
        assert torch.equal(batch[2], torch.ones(2, dtype=torch.float64))


@patch("modyn.supervisor.internal.triggers.trigger_datasets.fixed_keys_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.supervisor.internal.triggers.trigger_datasets.fixed_keys_dataset.grpc_connection_established",
    return_value=True,
)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_dataloader_dataset_multi_worker(test_insecure_channel, test_grpc_connection_established):
    if platform.system() == "Darwin":
        # On macOS, spawn is the default, which loses the mocks
        # Hence the test does not work on macOS, only on Linux.
        return

    fixed_keys_dataset = FixedKeysDataset(
        dataset_id=DATASET_ID,
        bytes_parser="def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')",
        serialized_transforms=[],
        storage_address=STORAGE_ADDR,
        trigger_id=TRIGGER_ID,
        keys=list(range(16)),
    )
    dataloader = torch.utils.data.DataLoader(fixed_keys_dataset, batch_size=4, num_workers=4)

    data = list(dataloader)
    data.sort(key=lambda batch_data: batch_data[0].min())

    batch_num = -1
    for batch_num, batch in enumerate(data):
        assert len(batch) == 3
        assert batch[0].tolist() == [4 * batch_num, 4 * batch_num + 1, 4 * batch_num + 2, 4 * batch_num + 3]
        assert torch.equal(
            batch[1], torch.Tensor([4 * batch_num, 4 * batch_num + 1, 4 * batch_num + 2, 4 * batch_num + 3])
        )
        assert torch.equal(batch[2], torch.ones(4, dtype=torch.float64))

    assert batch_num == 3
