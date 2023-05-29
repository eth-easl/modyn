import os

import pytest
import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.sample_then_batch_temporary_storage import (
    SampleThenBatchTemporaryStorage,
)


def test_init_accumulation():
    # batch size of 4, we want to extract 2 batches so 8 samples
    handler = SampleThenBatchTemporaryStorage(0, 0, 4, 50, 10000)

    handler.reset_temporary_storage()
    handler.accumulate([1, 2, 3, 4], torch.Tensor([1, 2, 3, 4]))
    handler.accumulate([11, 12, 13, 14], torch.Tensor([1, 2, 3, 4]))
    handler.accumulate([21, 22, 23, 24], torch.Tensor([1, 2, 3, 4]))
    handler.accumulate([31, 32, 33, 34], torch.Tensor([1, 2, 3, 4]))
    assert handler.normalizer == 40
    handler.end_accumulation()

    assert len(os.listdir(".tmp_scores")) == 1
    assert handler.number_of_samples_per_file == [16]

    assert handler.normalizer == 40 / 16
    assert sum(handler.grouped_samples_per_file) == 8

    samples_per_file = handler.grouped_samples_per_file
    samples = handler.get_samples_per_file(0)
    assert len(samples) == samples_per_file[0]
    selected_ids = [el[0] for el in samples]

    assert not os.path.isdir(".tmp_scores")
    assert set(selected_ids) <= {1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34}


def test_init_accumulation_limit():
    # batch size of 4, we want to extract 2 batches
    handler = SampleThenBatchTemporaryStorage(0, 0, 4, 50, 4)

    handler.reset_temporary_storage()
    handler.accumulate([1, 2, 3, 4], torch.Tensor([1, 2, 3, 4]))
    handler.accumulate([11, 12, 13, 14], torch.Tensor([1, 2, 3, 4]))
    handler.accumulate([21, 22, 23, 24], torch.Tensor([1, 2, 3, 4]))
    handler.accumulate([31, 32, 33, 34], torch.Tensor([1, 2, 3, 4]))
    assert len(os.listdir(".tmp_scores")) == 4
    assert handler.number_of_samples_per_file == [4, 4, 4, 4]
    assert handler.normalizer == 40
    handler.end_accumulation()

    assert handler.normalizer == 40 / 16
    assert sum(handler.grouped_samples_per_file) == 8

    samples_per_file = handler.grouped_samples_per_file
    selected_ids = []
    for file in range(4):
        samples = handler.get_samples_per_file(file)
        assert len(samples) == samples_per_file[file]
        selected_ids += [el[0] for el in samples]

    assert not os.path.isdir(".tmp_scores")
    assert set(selected_ids) <= {1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34}


def test_skewed_distribution():
    handler = SampleThenBatchTemporaryStorage(0, 0, 4, 50, 1000)

    # samples from even files are useless
    handler.reset_temporary_storage()
    handler.accumulate([1, 2, 3, 4], torch.Tensor([1, 2, 3, 4]))
    handler.accumulate([11, 12, 13, 14], torch.Tensor([0, 0, 0, 0]))
    handler.accumulate([21, 22, 23, 24], torch.Tensor([1, 2, 3, 4]))
    handler.accumulate([31, 32, 33, 34], torch.Tensor([0, 0, 0, 0]))
    assert handler.normalizer == 20
    handler.end_accumulation()

    assert len(handler.number_of_samples_per_file) == 1

    assert handler.normalizer == 20 / 16

    samples = handler.get_samples_per_file(0)
    selected_ids = [el[0] for el in samples]

    assert sorted(selected_ids) == [1, 2, 3, 4, 21, 22, 23, 24]


def test_restart_accumulation():
    handler = SampleThenBatchTemporaryStorage(0, 0, 4, 50, 1000)

    # samples from even files are useless
    handler.reset_temporary_storage()
    handler.accumulate([1, 2, 3, 4], torch.Tensor([1, 2, 3, 4]))
    handler.accumulate([11, 12, 13, 14], torch.Tensor([1, 2, 3, 4]))
    handler.accumulate([21, 22, 23, 24], torch.Tensor([1, 2, 3, 4]))
    handler.accumulate([31, 32, 33, 34], torch.Tensor([1, 2, 3, 4]))
    assert handler.normalizer == 40
    handler.end_accumulation()

    assert len(os.listdir(".tmp_scores")) == 1
    assert handler.number_of_samples_per_file == [16]

    assert handler.normalizer == 40 / 16
    assert sum(handler.grouped_samples_per_file) == 8

    handler.reset_temporary_storage()
    assert handler.normalizer == 0
    assert not handler.number_of_samples_per_file
    assert not os.path.isdir(".tmp_scores")
    handler.accumulate([101, 102, 103, 104], torch.Tensor([10, 3, 3, 4]))
    handler.accumulate([111, 112, 113, 114], torch.Tensor([10, 3, 3, 4]))
    handler.accumulate([121, 122, 123, 124], torch.Tensor([10, 3, 3, 4]))
    assert handler.normalizer == 60
    handler.end_accumulation()

    assert len(os.listdir(".tmp_scores")) == 1
    assert handler.number_of_samples_per_file == [12]

    assert handler.normalizer == 60 / 12
    assert sum(handler.grouped_samples_per_file) == 6

    samples = handler.get_samples_per_file(0)
    selected_ids = [el[0] for el in samples]
    assert all(el > 100 for el in selected_ids)

    with pytest.raises(AssertionError):
        handler.get_samples_per_file(1)


def test_restart_accumulation_limited():
    handler = SampleThenBatchTemporaryStorage(0, 0, 4, 50, 4)

    # samples from even files are useless
    handler.reset_temporary_storage()
    handler.accumulate([1, 2, 3, 4], torch.Tensor([1, 2, 3, 4]))
    handler.accumulate([11, 12, 13, 14], torch.Tensor([1, 2, 3, 4]))
    handler.accumulate([21, 22, 23, 24], torch.Tensor([1, 2, 3, 4]))
    handler.accumulate([31, 32, 33, 34], torch.Tensor([1, 2, 3, 4]))
    assert len(os.listdir(".tmp_scores")) == 4

    assert handler.number_of_samples_per_file == [4, 4, 4, 4]
    assert handler.normalizer == 40
    handler.end_accumulation()

    assert len(os.listdir(".tmp_scores")) == 4
    assert handler.normalizer == 40 / 16
    assert sum(handler.grouped_samples_per_file) == 8

    handler.reset_temporary_storage()
    assert handler.normalizer == 0
    assert not handler.number_of_samples_per_file
    assert not os.path.isdir(".tmp_scores")
    handler.accumulate([101, 102, 103, 104], torch.Tensor([10, 3, 3, 4]))
    handler.accumulate([111, 112, 113, 114], torch.Tensor([10, 3, 3, 4]))
    handler.accumulate([121, 122, 123, 124], torch.Tensor([10, 3, 3, 4]))
    assert len(os.listdir(".tmp_scores")) == 3

    assert handler.number_of_samples_per_file == [4, 4, 4]
    assert handler.normalizer == 60
    handler.end_accumulation()

    assert handler.normalizer == 60 / 12
    assert sum(handler.grouped_samples_per_file) == 6

    selected_ids = []
    for file in range(3):
        samples = handler.get_samples_per_file(file)
        selected_ids += [el[0] for el in samples]
        assert all(el > 100 for el in selected_ids)

    with pytest.raises(AssertionError):
        handler.get_samples_per_file(3)
