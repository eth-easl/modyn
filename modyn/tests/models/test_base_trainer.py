# pylint: disable=unused-argument
from typing import Optional
import torch
import os


class DummyDataset(torch.utils.data.dataset.Dataset):
    def __init__(self) -> None:
        super().__init__()


class DummyStatefulObject():
    def __init__(self) -> None:
        self._state = {}

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self._state = state


test_dataloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)


class DummyTrainer:

    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        device: int,
        checkpoint_path: str,
        checkpoint_interval: int
    ):

        super().__init__(train_loader, val_loader, device, checkpoint_path, checkpoint_interval)
        self._model = DummyStatefulObject()
        self._optimizer = DummyStatefulObject()

    def train_one_iteration(self, iteration, batch):
        pass


def test_save_checkpoint():

    trainer = DummyTrainer(
        test_dataloader,
        None,
        0,
        "checkpoint_test",
        10
    )
    trainer.save_checkpoint(iteration=10)

    assert os.path.exists("checkpoint_test/model_10.pt")
    saved_dict = torch.load("checkpoint_test/model_10.pt")

    assert saved_dict == {
        'model': {},
        'optimizer': {}
    }


def test_load_checkpoint():

    trainer = DummyTrainer(
        test_dataloader,
        None,
        0,
        "checkpoint_test",
        10
    )

    dict_to_save = {
        'model': {'a': 1.0},
        'optimizer': {'a': 2.0}
    }
    torch.save(dict_to_save, "checkpoint_test/model_20.pt")

    trainer.load_checkpoint("checkpoint_test/model_20.pt")
    assert trainer._model.state_dict() == dict_to_save['model']
    assert trainer._optimizer.state_dict() == dict_to_save['optimizer']
