from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader

from modyn.models.yearbooknet.yearbooknet import YearbookNet
from modyn.supervisor.internal.triggers.drift.embedding.embeddings import get_embeddings
from modyn.supervisor.internal.triggers.utils.model.manager import StatefulModel


@pytest.fixture
def mock_stateful_model() -> StatefulModel:
    mock_manager = MagicMock(spec=StatefulModel)

    mock_manager.device = torch.device("cpu")
    mock_manager.device_type = "cpu"
    mock_manager.amp = False

    dummy_model = YearbookNet({"num_classes": 2, "num_input_channels": 3}, "cpu", False)

    # Overwrite the mock's internal model with the dummy model
    mock_manager._model = dummy_model
    return mock_manager


@pytest.fixture
def mock_dataloader() -> DataLoader:
    data = torch.randn(50, 3, 32, 32)
    labels = torch.randint(0, 2, (50,)).float()
    dataloader = DataLoader(list(zip(labels, data)), batch_size=1)
    return dataloader


def test_get_embeddings(mock_stateful_model: StatefulModel, mock_dataloader: DataLoader) -> None:
    embeddings = get_embeddings(mock_stateful_model, mock_dataloader)

    assert embeddings.shape == (50, 32)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings is not None
