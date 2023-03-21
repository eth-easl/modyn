import os
import pathlib

import pytest
import torch
from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.selector.internal.selector_strategies.loss_downsample import LossDownsample
from modyn.backend.selector.internal.selector_strategies.remote_loss_downsample import RemoteLossDownsampler

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"


def get_minimal_modyn_config():
    return {
        "metadata_database": {
            "drivername": "sqlite",
            "username": "",
            "password": "",
            "host": "",
            "port": "0",
            "database": f"{database_path}",
        },
        "selector": {"insertion_threads": 8},
    }


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()
    yield

    os.remove(database_path)


def test_init():
    # Test init works
    strat = LossDownsample(
        {"limit": -1, "reset_after_trigger": False, "presampling_ratio": 80}, get_minimal_modyn_config(), 42, 1000, 10
    )

    assert strat.downsampled_batch_size == 10
    assert strat._pipeline_id == 42
    assert isinstance(strat.get_downsampling_strategy(), str)


def test_command():
    # Test init works
    strat = LossDownsample(
        {"limit": -1, "reset_after_trigger": False, "presampling_ratio": 80}, get_minimal_modyn_config(), 42, 1000, 10
    )

    cmd = strat.get_downsampling_strategy()
    assert isinstance(cmd, str)
    assert (
        cmd
        == 'RemoteLossDownsampler(self._model, 10, criterion_func(**training_info.criterion_dict, reduction="none"))'
    )


def test_sample_shape():
    model = torch.nn.Linear(10, 2)
    downsampled_batch_size = 5
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    sampler = RemoteLossDownsampler(model, downsampled_batch_size, per_sample_loss_fct)

    data = torch.randn(8, 10)
    target = torch.randint(2, size=(8,))

    sampled_data, weights, sampled_target = sampler.sample(data, target)

    assert sampled_data.shape[0] == downsampled_batch_size
    assert sampled_data.shape[1] == data.shape[1]
    assert weights.shape[0] == downsampled_batch_size
    assert sampled_target.shape[0] == downsampled_batch_size


def test_sample_weights_sum_to_one():
    model = torch.nn.Linear(10, 2)
    downsampled_batch_size = 5
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    sampler = RemoteLossDownsampler(model, downsampled_batch_size, per_sample_loss_fct)

    data = torch.randn(8, 10)
    target = torch.randint(2, size=(8,))
    _, weights, _ = sampler.sample(data, target)

    assert weights.sum() > 0


# Create a model that always predicts the same class
class AlwaysZeroModel(torch.nn.Module):
    def forward(self, data):
        return torch.zeros(data.shape[0], 2)


def test_sample_loss_dependent_sampling():
    model = AlwaysZeroModel()
    downsampled_batch_size = 5
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    sampler = RemoteLossDownsampler(model, downsampled_batch_size, per_sample_loss_fct)

    # Create a target with two classes, where half have a true label of 0 and half have a true label of 1
    target = torch.cat([torch.zeros(4), torch.ones(4)]).long()

    # Create a data tensor with four points that have a loss of zero and four points that have a non-zero loss
    data = torch.cat([torch.randn(4, 10), torch.randn(4, 10)], dim=0)
    output = torch.zeros(8, 2)
    output[:4, 0] = 1  # Set the scores for the first four points to zero
    output[4:, 1] = 1  # Set the scores for the last four points to one

    _, _, sampled_target = sampler.sample(data, target)

    # Assert that no points with a loss of zero were selected
    assert (sampled_target == 0).sum() > 0
    assert (sampled_target > 0).sum() > 0
