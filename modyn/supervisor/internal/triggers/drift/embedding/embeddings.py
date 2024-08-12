import torch
from torch.utils.data import DataLoader

from modyn.supervisor.internal.triggers.embedding_encoder import EmbeddingEncoder


def get_embeddings(embedding_encoder: EmbeddingEncoder, dataloader: DataLoader) -> torch.Tensor:
    """
    input: embedding_encoder with downloaded model
    output: embeddings Tensor
    """
    assert embedding_encoder._model is not None
    all_embeddings: torch.Tensor | None = None

    embedding_encoder._model.model.eval()
    embedding_encoder._model.model.embedding_recorder.start_recording()

    with torch.no_grad():
        for batch in dataloader:
            data: torch.Tensor | dict
            if isinstance(batch[1], torch.Tensor):
                data = batch[1].to(embedding_encoder.device)
            elif isinstance(batch[1], dict):
                data: dict[str, torch.Tensor] = {}  # type: ignore[no-redef]
                for name, tensor in batch[1].items():
                    data[name] = tensor.to(embedding_encoder.device)
            else:
                raise ValueError(f"data type {type(batch[1])} not supported")

            with torch.autocast(embedding_encoder.device_type, enabled=embedding_encoder.amp):
                embedding_encoder._model.model(data)
                embeddings = embedding_encoder._model.model.embedding_recorder.embedding
                if all_embeddings is None:
                    all_embeddings = embeddings
                else:
                    all_embeddings = torch.cat((all_embeddings, embeddings), 0)

    embedding_encoder._model.model.embedding_recorder.end_recording()

    return all_embeddings
