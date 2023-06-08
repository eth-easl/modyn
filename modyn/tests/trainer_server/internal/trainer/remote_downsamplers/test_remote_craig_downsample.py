import pytest
import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsample_strategy import (
    get_tensors_subset,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_craig_downsample import RemoteCRAIGDownsampling


def test_sample_shape_ce():
    model = torch.nn.Linear(10, 3)
    downsampled_batch_size = 5
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {
        "downsampled_batch_size": downsampled_batch_size,
        "sample_then_batch": False,
        "selection_type": "Supervised",
    }
    sampler = RemoteCRAIGDownsampling(0, 0, 0, params_from_selector, per_sample_loss_fct)

    data = torch.randn(8, 10)
    target = torch.randint(2, size=(8,))
    ids = list(range(0, 8))
    forward_outputs = model(data)

    downsampled_indexes, weights = sampler.batch_then_sample(forward_outputs, target)

    assert downsampled_indexes.shape[0] == downsampled_batch_size
    assert weights.shape[0] == downsampled_batch_size

    sampled_data, sampled_target, sampled_ids = get_tensors_subset(downsampled_indexes, data, target, ids)

    assert weights.shape[0] == sampled_target.shape[0]
    assert sampled_data.shape[0] == downsampled_batch_size
    assert sampled_data.shape[1] == data.shape[1]
    assert weights.shape[0] == downsampled_batch_size
    assert sampled_target.shape[0] == downsampled_batch_size
    assert set(sampled_ids) <= set(range(8))


def test_init():
    params_from_selector_bts = {
        "downsampled_batch_size": 4,
        "sample_then_batch": False,
        "selection_type": "Supervised",
    }
    params_from_selector_stb = {
        "downsampled_batch_ratio": 4,
        "sample_then_batch": True,
        "selection_type": "Supervised",
    }
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    # bts
    sampler = RemoteCRAIGDownsampling(0, 0, 0, params_from_selector_bts, per_sample_loss_fct)

    with pytest.raises(AssertionError):
        sampler.setup_sample_then_batch()

    with pytest.raises(AssertionError):
        sampler.accumulate_sample_then_batch(None, None, None)

    with pytest.raises(AssertionError):
        sampler.end_sample_then_batch()

    with pytest.raises(AssertionError):
        sampler.samples_available()

    with pytest.raises(AssertionError):
        sampler.get_samples()

    # stb
    sampler = RemoteCRAIGDownsampling(0, 0, 0, params_from_selector_stb, per_sample_loss_fct)

    with pytest.raises(AssertionError):
        sampler.batch_then_sample(None, None)


def test_stb():
    batch_size = 26
    model = torch.nn.Linear(10, 3)
    params_from_selector_stb = {
        "downsampled_batch_ratio": 50,
        "sample_then_batch": True,
        "selection_type": "Supervised",
    }
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    sampler = RemoteCRAIGDownsampling(0, 0, batch_size, params_from_selector_stb, per_sample_loss_fct)

    sampler.setup_sample_then_batch()

    batch_counter = 0

    for _ in range(5):
        data = torch.randn(batch_size, 10)
        target = torch.randint(2, size=(batch_size,))
        # to decouple the mapping counter/index
        ids = list(range(100 * batch_counter, batch_counter * 100 + batch_size))
        forward_outputs = model(data)
        sampler.accumulate_sample_then_batch(forward_outputs, target, ids)
        batch_counter += 1

    # to simulate 2 dataloaders

    # last, smaller, batch
    size = 10
    data = torch.randn(size, 10)
    target = torch.randint(2, size=(size,))
    # to decouple the mapping counter/index
    ids = list(range(100 * batch_counter, batch_counter * 100 + size))
    forward_outputs = model(data)
    sampler.accumulate_sample_then_batch(forward_outputs, target, ids)
    batch_counter += 1

    # last, smaller, batch
    size = 8
    data = torch.randn(size, 10)
    target = torch.randint(2, size=(size,))
    # to decouple the mapping counter/index
    ids = list(range(100 * batch_counter, batch_counter * 100 + size))
    forward_outputs = model(data)
    sampler.accumulate_sample_then_batch(forward_outputs, target, ids)
    batch_counter += 1

    sampler.end_sample_then_batch()

    assert sampler.samples_available()

    sampled_points = sampler.get_samples()

    assert len(sampled_points) == 0.5 * 26 * 5 + 5 + 4

    for index, weight in sampled_points:
        assert weight > 0
        # each id is built as list(range(100*batch_counter, batch_counter*100 + batch_size))
        assert index % 100 < batch_size

    assert not sampler.samples_available()
