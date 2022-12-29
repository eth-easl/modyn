import pytest
from unittest.mock import patch

from modyn.backend.selector.mock_selector_server import GetSamplesResponse, MockSelectorServer
from modyn.gpu_node.data.online_dataset import OnlineDataset
from modyn.storage.mock_storage_server import GetResponse, MockStorageServer

@patch.object(MockSelectorServer, 'get_sample_keys', return_value=GetSamplesResponse(training_samples_subset=[1,2,3]))
def test_get_keys_from_selector(test_get_sample_keys):

    online_dataset = OnlineDataset(training_id=1)
    assert online_dataset._get_keys_from_selector(0) == [1,2,3]


@patch.object(MockStorageServer, 'Get', return_value=GetResponse(value=["sample0", "sample1"]))
def test_get_data_from_storage(test_get):

    online_dataset = OnlineDataset(training_id=1)
    assert online_dataset._get_data_from_storage([]) == ["sample0", "sample1"]

@patch.object(OnlineDataset, '_process', return_value=list(range(10)))
@patch.object(OnlineDataset, '_get_data_from_storage', return_value=list(range(10)))
@patch.object(OnlineDataset, '_get_keys_from_selector', return_value=list(range(10)))
def test_dataset_iter(test_process, test_get_data, test_get_keys):

    online_dataset = OnlineDataset(training_id=1)
    dataset_iter = iter(online_dataset)
    assert list(dataset_iter) == list(range(10))