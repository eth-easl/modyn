import torch
from torch.utils.data import DataLoader

from modyn.supervisor.internal.triggers.utils.model.manager import ModelManager


def get_embeddings(model_manager: ModelManager, dataloader: DataLoader) -> torch.Tensor:
    """
    input: embedding_encoder with downloaded model
    output: embeddings Tensor
    """
    assert model_manager._model is not None
    all_embeddings: torch.Tensor | None = None

    model_manager._model.model.eval()
    model_manager._model.model.embedding_recorder.start_recording()

    with torch.no_grad():
        for batch in dataloader:
            data: torch.Tensor | dict
            if isinstance(batch[1], torch.Tensor):
                data = batch[1].to(model_manager.device)
            elif isinstance(batch[1], dict):
                data: dict[str, torch.Tensor] = {}  # type: ignore[no-redef]
                for name, tensor in batch[1].items():
                    data[name] = tensor.to(model_manager.device)
            else:
                raise ValueError(f"data type {type(batch[1])} not supported")

            with torch.autocast(model_manager.device_type, enabled=model_manager.amp):
                model_manager._model.model(data)
                embeddings = model_manager._model.model.embedding_recorder.embedding
                if all_embeddings is None:
                    all_embeddings = embeddings
                else:
                    all_embeddings = torch.cat((all_embeddings, embeddings), 0)

    model_manager._model.model.embedding_recorder.end_recording()

    return all_embeddings
