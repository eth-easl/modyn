# pylint: disable=unused-argument
from unittest.mock import patch

import pytest
import torch
from modyn.trainer_server.internal.dataset.online_dataset import OnlineDataset
from modyn.trainer_server.internal.mocks.mock_selector_server import GetSamplesResponse, MockSelectorServer
from modyn.trainer_server.internal.mocks.mock_storage_server import GetResponse, MockStorageServer
from torchvision import transforms


@patch.object(
    MockSelectorServer,
    "get_sample_keys",
    return_value=GetSamplesResponse(training_samples_subset=[1, 2, 3]),
)
def test_get_keys_from_selector(test_get_sample_keys):

    online_dataset = OnlineDataset(
        training_id=1,
        dataset_id="MNIST",
        serialized_transforms=[],
        train_until_sample_id="new",
    )
    assert online_dataset._get_keys_from_selector(0) == [1, 2, 3]


@patch.object(
    MockStorageServer,
    "Get",
    return_value=GetResponse(data=["sample0", "sample1"], labels=[0, 1]),
)
def test_get_data_from_storage(test_get):

    online_dataset = OnlineDataset(
        training_id=1,
        dataset_id="MNIST",
        serialized_transforms=[],
        train_until_sample_id="new",
    )
    assert online_dataset._get_data_from_storage([]) == (["sample0", "sample1"], [0, 1])


@pytest.mark.parametrize(
    "serialized_transforms,transforms_list",
    [
        pytest.param(
            [
                "transforms.RandomResizedCrop(224)",
                "transforms.RandomHorizontalFlip()",
                "transforms.ToTensor()",
                "transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
            ],
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ],
        )
    ],
)
def test_deserialize_torchvision_transforms(serialized_transforms, transforms_list):

    online_dataset = OnlineDataset(
        training_id=1,
        dataset_id="MNIST",
        serialized_transforms=serialized_transforms,
        train_until_sample_id="new",
    )
    online_dataset._deserialize_torchvision_transforms()
    assert isinstance(online_dataset._transform.transforms, list)
    for transform1, transform2 in zip(
        online_dataset._transform.transforms, transforms_list
    ):
        assert transform1.__dict__ == transform2.__dict__


@patch.object(
    OnlineDataset, "_get_data_from_storage", return_value=(list(range(10)), [1] * 10)
)
@patch.object(OnlineDataset, "_get_keys_from_selector", return_value=[])
def test_dataset_iter(test_get_data, test_get_keys):

    online_dataset = OnlineDataset(
        training_id=1,
        dataset_id="MNIST",
        serialized_transforms=[],
        train_until_sample_id="new",
    )
    dataset_iter = iter(online_dataset)
    all_data = list(dataset_iter)
    assert [x[0] for x in all_data] == list(range(10))
    assert [x[1] for x in all_data] == [1] * 10


@patch.object(
    OnlineDataset, "_get_data_from_storage", return_value=([0] * 16, [1] * 16)
)
@patch.object(OnlineDataset, "_get_keys_from_selector", return_value=[])
def test_dataloader_dataset(test_get_data, test_get_keys):

    online_dataset = OnlineDataset(
        training_id=1,
        dataset_id="MNIST",
        serialized_transforms=[],
        train_until_sample_id="new",
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4)
    for batch in dataloader:
        assert len(batch) == 2
        assert torch.equal(batch[0], torch.zeros(4, dtype=int))
        assert torch.equal(batch[1], torch.ones(4, dtype=int))
