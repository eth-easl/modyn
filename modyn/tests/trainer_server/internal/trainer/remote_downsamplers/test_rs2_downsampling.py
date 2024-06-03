import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    get_tensors_subset,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_rs2_downsampling import (
    RemoteRS2Downsampling,
)


def test_init():
    pipeline_id = 0
    trigger_id = 0
    batch_size = 32
    params_from_selector = {"replacement": True, "downsampling_ratio": 50}
    per_sample_loss = None
    device = "cpu"

    downsampler = RemoteRS2Downsampling(
        pipeline_id, trigger_id, batch_size, params_from_selector, per_sample_loss, device
    )

    assert downsampler.pipeline_id == pipeline_id
    assert downsampler.trigger_id == trigger_id
    assert downsampler.batch_size == batch_size
    assert downsampler.device == device
    assert downsampler.forward_required == False
    assert downsampler.supports_bts == False
    assert downsampler._all_sample_ids == []
    assert downsampler._subsets == []
    assert downsampler._current_subset == -1
    assert downsampler._with_replacement == params_from_selector["replacement"]
    assert downsampler._max_subset == -1
    assert downsampler._first_epoch == True


def test_inform_samples():
    pipeline_id = 0
    trigger_id = 0
    batch_size = 32
    params_from_selector = {"replacement": True, "downsampling_ratio": 50}
    per_sample_loss = None
    device = "cpu"

    downsampler = RemoteRS2Downsampling(
        pipeline_id, trigger_id, batch_size, params_from_selector, per_sample_loss, device
    )

    sample_ids = [1, 2, 3, 4, 5]
    forward_output = torch.randn(5, 10)
    target = torch.randint(0, 10, (5,))

    downsampler.inform_samples(sample_ids, forward_output, target)

    assert downsampler._all_sample_ids == sample_ids
    downsampler.inform_samples(sample_ids, forward_output, target)
    assert downsampler._all_sample_ids == 2 * sample_ids
    # Now it should not change anymore
    downsampler.select_points()
    downsampler.inform_samples(sample_ids, forward_output, target)
    assert set(downsampler._all_sample_ids) == set(sample_ids)
    assert len(downsampler._all_sample_ids) == 2* len(sample_ids)

def test_multiple_epochs_with_replacement():
    pipeline_id = 0
    trigger_id = 0
    batch_size = 32
    params_from_selector = {"replacement": True, "downsampling_ratio": 50}
    per_sample_loss = None
    device = "cpu"

    downsampler = RemoteRS2Downsampling(
        pipeline_id, trigger_id, batch_size, params_from_selector, per_sample_loss, device
    )
    with torch.inference_mode(mode=(not downsampler.requires_grad)):
        sample_ids = list(range(10))
        data = torch.randn(10, 10)
        target = torch.randint(0, 10, (10,))

        for _ in range(3):
            downsampler.inform_samples(sample_ids, data, target)
            selected_ids, weights = downsampler.select_points()
            sampled_data, sampled_target = get_tensors_subset(selected_ids, data, target, sample_ids)

            assert len(set(selected_ids)) == 5
            assert weights.shape == (5,)
            assert all(idx in sample_ids for idx in selected_ids)
            assert sampled_data.shape == (5, 10)
            assert sampled_target.shape == (5,)


def test_multiple_epochs_without_replacement():
    pipeline_id = 0
    trigger_id = 0
    batch_size = 32
    params_from_selector = {"replacement": False, "downsampling_ratio": 50}
    per_sample_loss = None
    device = "cpu"

    downsampler = RemoteRS2Downsampling(
        pipeline_id, trigger_id, batch_size, params_from_selector, per_sample_loss, device
    )
    with torch.inference_mode(mode=(not downsampler.requires_grad)):

        sample_ids = list(range(10))
        data = torch.randn(10, 10)
        target = torch.randint(0, 10, (10,))

        # Epoch 1
        downsampler.inform_samples(sample_ids, data, target)
        epoch1_ids, weights = downsampler.select_points()
        sampled_data, sampled_target = get_tensors_subset(epoch1_ids, data, target, sample_ids)

        assert len(set(epoch1_ids)) == 5
        assert weights.shape == (5,)
        assert all(idx in sample_ids for idx in epoch1_ids)
        assert sampled_data.shape == (5, 10)
        assert sampled_target.shape == (5,)

        # Epoch 2
        downsampler.inform_samples(sample_ids, data, target)
        epoch2_ids, weights = downsampler.select_points()
        sampled_data, sampled_target = get_tensors_subset(epoch2_ids, data, target, sample_ids)

        assert len(set(epoch2_ids)) == 5
        assert weights.shape == (5,)
        assert all(idx in sample_ids for idx in epoch2_ids)
        assert not any(idx in epoch1_ids for idx in epoch2_ids) # No overlap across epochs
        assert sampled_data.shape == (5, 10)
        assert sampled_target.shape == (5,)

        # Epoch 3
        downsampler.inform_samples(sample_ids, data, target)
        epoch3_ids, weights = downsampler.select_points()
        sampled_data, sampled_target = get_tensors_subset(epoch3_ids, data, target, sample_ids)

        assert len(set(epoch3_ids)) == 5
        assert weights.shape == (5,)
        assert all(idx in sample_ids for idx in epoch3_ids)
        assert all(idx in epoch1_ids or idx in epoch2_ids for idx in epoch3_ids) # There needs to be overlap now
        assert any(idx not in epoch1_ids for idx in epoch3_ids) # but (with very high probability, this might be flaky lets see) there is some difference
        assert sampled_data.shape == (5, 10)
        assert sampled_target.shape == (5,)


def test_multiple_epochs_without_replacement_leftover_data():
    pipeline_id = 0
    trigger_id = 0
    batch_size = 32
    params_from_selector = {"replacement": False, "downsampling_ratio": 40}
    per_sample_loss = None
    device = "cpu"

    downsampler = RemoteRS2Downsampling(
        pipeline_id, trigger_id, batch_size, params_from_selector, per_sample_loss, device
    )
    with torch.inference_mode(mode=(not downsampler.requires_grad)):
        sample_ids = list(range(10))
        data = torch.randn(10, 10)
        target = torch.randint(0, 10, (10,))

        for _ in range(3):
            downsampler.inform_samples(sample_ids, data, target)

            selected_ids, weights = downsampler.select_points()
            sampled_data, sampled_target = get_tensors_subset(selected_ids, data, target, sample_ids)
            assert len(set(selected_ids)) == 4
            assert weights.shape == (4,)
            assert sampled_data.shape == (4, 10)
            assert sampled_target.shape == (4,)

            assert all(idx in sample_ids for idx in selected_ids)
            assert len(set(selected_ids)) == len(selected_ids)
