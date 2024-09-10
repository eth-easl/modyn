import pytest
import torch

from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    get_tensors_subset,
)


def sample_data():
    # Create sample data for testing
    data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    target = torch.tensor([0, 1, 0])
    sample_ids = [102, 132, 154]
    selected_indexes = [132, 154]
    return selected_indexes, data, target, sample_ids


def test_get_tensors_subset():
    selected_indexes, data, target, sample_ids = sample_data()

    sub_data, sub_target = get_tensors_subset(selected_indexes, data, target, sample_ids)

    # Check if the selected data and target tensors have the correct shape and values
    assert sub_data.shape == (len(selected_indexes), data.shape[1])
    assert torch.equal(sub_data, data[[1, 2]])

    assert sub_target.shape == (len(selected_indexes),)
    assert torch.equal(sub_target, target[[1, 2]])


def sample_data_dict():
    # Create sample data for testing
    data = {
        "feature1": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "feature2": torch.tensor([[10, 11], [12, 13], [14, 15]]),
    }
    target = torch.tensor([0, 1, 0])
    sample_ids = [102, 132, 154]
    selected_indexes = [132, 154]
    return selected_indexes, data, target, sample_ids


def test_get_tensors_subset_empty():
    selected_indexes, data, target, sample_ids = sample_data()

    # Test when selected_indexes is an empty list
    selected_indexes = []
    sub_data, sub_target = get_tensors_subset(selected_indexes, data, target, sample_ids)

    # Check if the selected data and target tensors are empty
    assert sub_data.shape == (0, data.shape[1])
    assert sub_data.numel() == 0

    assert sub_target.shape == (0,)
    assert sub_target.numel() == 0


def test_get_tensors_subset_invalid_index():
    selected_indexes, data, target, sample_ids = sample_data()

    # Test when selected_indexes contains an invalid index
    selected_indexes.append(999)

    with pytest.raises(KeyError):
        get_tensors_subset(selected_indexes, data, target, sample_ids)


def test_get_tensors_subset_dict():
    selected_indexes, data, target, sample_ids = sample_data_dict()

    # Convert the dictionary data to a dictionary of tensors
    data_tensors = {key: torch.tensor(value) for key, value in data.items()}

    sub_data, sub_target = get_tensors_subset(selected_indexes, data_tensors, target, sample_ids)

    # Check if the selected data and target tensors have the correct shape and values
    assert sub_data["feature1"].shape == (len(selected_indexes), data["feature1"].shape[1])
    assert torch.equal(sub_data["feature1"], data["feature1"][[1, 2]])

    assert sub_data["feature2"].shape == (len(selected_indexes), data["feature2"].shape[1])
    assert torch.equal(sub_data["feature2"], data["feature2"][[1, 2]])

    assert sub_target.shape == (len(selected_indexes),)
    assert torch.equal(sub_target, target[[1, 2]])


def test_get_tensors_subset_empty_dict():
    selected_indexes, data, target, sample_ids = sample_data_dict()

    # Convert the dictionary data to a dictionary of tensors
    data_tensors = {key: torch.tensor(value) for key, value in data.items()}

    # no selected indexes
    selected_indexes = []

    sub_data, sub_target = get_tensors_subset(selected_indexes, data_tensors, target, sample_ids)

    # Check if the selected data and target tensors have the correct shape and values
    assert sub_data["feature1"].shape == (0, data["feature1"].shape[1])
    assert sub_data["feature1"].numel() == 0

    assert sub_data["feature2"].shape == (0, data["feature2"].shape[1])
    assert sub_data["feature2"].numel() == 0

    assert sub_target.shape == (0,)
    assert sub_target.numel() == 0
