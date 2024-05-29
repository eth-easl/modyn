import unittest

import torch
from modyn.trainer_server.internal.trainer.batch_accumulator import BatchAccumulator


class TestBatchAccumulator(unittest.TestCase):
    def setUp(self):
        self.accumulation_period = 3
        self.target_device = "cpu"
        self.accumulator = BatchAccumulator(self.accumulation_period, self.target_device)

    def test_inform_batch_tensor(self):
        data = torch.randn(2, 3)
        sample_ids = [1, 2]
        target = torch.randn(2)
        weights = torch.randn(2)

        ready_to_accumulate = self.accumulator.inform_batch(data, sample_ids, target, weights)
        self.assertFalse(ready_to_accumulate)
        self.assertEqual(len(self.accumulator._accumulation_buffer), 1)

    def test_inform_batch_dict(self):
        data = {"input1": torch.randn(2, 3), "input2": torch.randn(2, 4)}
        sample_ids = [1, 2]
        target = torch.randn(2)
        weights = torch.randn(2)

        ready_to_accumulate = self.accumulator.inform_batch(data, sample_ids, target, weights)
        self.assertFalse(ready_to_accumulate)
        self.assertEqual(len(self.accumulator._accumulation_buffer), 1)

    def test_inform_batch_accumulation_period(self):
        data = torch.randn(2, 3)
        sample_ids = [1, 2]
        target = torch.randn(2)
        weights = torch.randn(2)

        for i in range(self.accumulation_period):
            ready_to_accumulate = self.accumulator.inform_batch(data, sample_ids, target, weights)
            if i < self.accumulation_period - 1:
                self.assertFalse(ready_to_accumulate)
            else:
                self.assertTrue(ready_to_accumulate)

        self.assertEqual(len(self.accumulator._accumulation_buffer), self.accumulation_period)

    def test_get_accumulated_batch_tensor(self):
        data = torch.randn(2, 3)
        sample_ids = [1, 2]
        target = torch.randn(2)
        weights = torch.randn(2)

        for _ in range(self.accumulation_period):
            self.accumulator.inform_batch(data, sample_ids, target, weights)

        (
            accumulated_data,
            accumulated_sample_ids,
            accumulated_target,
            accumulated_weights,
        ) = self.accumulator.get_accumulated_batch()

        self.assertIsInstance(accumulated_data, torch.Tensor)
        self.assertEqual(accumulated_data.shape, (6, 3))
        self.assertEqual(len(accumulated_sample_ids), 6)
        self.assertEqual(accumulated_target.shape, (6,))
        self.assertEqual(accumulated_weights.shape, (6,))
        self.assertEqual(len(self.accumulator._accumulation_buffer), 0)

    def test_get_accumulated_batch_dict(self):
        data = {"input1": torch.randn(2, 3), "input2": torch.randn(2, 4)}
        sample_ids = [1, 2]
        target = torch.randn(2)
        weights = torch.randn(2)

        for _ in range(self.accumulation_period):
            self.accumulator.inform_batch(data, sample_ids, target, weights)

        (
            accumulated_data,
            accumulated_sample_ids,
            accumulated_target,
            accumulated_weights,
        ) = self.accumulator.get_accumulated_batch()

        self.assertIsInstance(accumulated_data, dict)
        self.assertEqual(accumulated_data["input1"].shape, (6, 3))
        self.assertEqual(accumulated_data["input2"].shape, (6, 4))
        self.assertEqual(len(accumulated_sample_ids), 6)
        self.assertEqual(accumulated_target.shape, (6,))
        self.assertEqual(accumulated_weights.shape, (6,))
        self.assertEqual(len(self.accumulator._accumulation_buffer), 0)

    def test_get_accumulated_batch_tensor_multi_round(self):
        data = [
            torch.Tensor([[1, 2, 3], [3, 4, 5]]),
            torch.Tensor([[6, 7, 8], [9, 10, 11]]),
            torch.Tensor([[12, 13, 14], [15, 16, 17]]),
        ]
        sample_ids = [[1, 2], [3, 4], [5, 6]]
        target = torch.randn(2)
        weights = torch.randn(2)

        for i in range(self.accumulation_period):
            self.accumulator.inform_batch(data[i], sample_ids[i], target, weights)

        (
            accumulated_data,
            accumulated_sample_ids,
            accumulated_target,
            accumulated_weights,
        ) = self.accumulator.get_accumulated_batch()

        self.assertIsInstance(accumulated_data, torch.Tensor)
        self.assertEqual(accumulated_data.shape, (6, 3))
        self.assertEqual(len(accumulated_sample_ids), 6)
        self.assertEqual(accumulated_target.shape, (6,))
        self.assertEqual(accumulated_weights.shape, (6,))
        self.assertEqual(len(self.accumulator._accumulation_buffer), 0)

        assert torch.allclose(
            accumulated_data, torch.Tensor([[1, 2, 3], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17],  ])
        )
        self.assertEqual(accumulated_sample_ids, [1,2,3,4,5,6])

        # Start another round

        data = torch.randn(2, 3)
        sample_ids = [1, 2]
        target = torch.randn(2)
        weights = torch.randn(2)

        for _ in range(self.accumulation_period):
            self.accumulator.inform_batch(data, sample_ids, target, weights)

        (
            accumulated_data,
            accumulated_sample_ids,
            accumulated_target,
            accumulated_weights,
        ) = self.accumulator.get_accumulated_batch()

        self.assertIsInstance(accumulated_data, torch.Tensor)
        self.assertEqual(accumulated_data.shape, (6, 3))
        self.assertEqual(len(accumulated_sample_ids), 6)
        self.assertEqual(accumulated_target.shape, (6,))
        self.assertEqual(accumulated_weights.shape, (6,))
        self.assertEqual(len(self.accumulator._accumulation_buffer), 0)
