import torch
from torch.utils.data import DataLoader

from modyn.supervisor.internal.triggers.utils.model.stateful_model import StatefulModel


def get_embeddings(stateful_model: StatefulModel, dataloader: DataLoader) -> torch.Tensor:
    """
    input: embedding_encoder with downloaded model
    output: embeddings Tensor
    """
    assert stateful_model.model is not None
    all_embeddings: torch.Tensor | None = None

    stateful_model.model.model.eval()
    stateful_model.model.model.embedding_recorder.start_recording()

    with torch.no_grad():
        for batch in dataloader:
            data: torch.Tensor | dict
            if isinstance(batch[1], torch.Tensor):
                data = batch[1].to(stateful_model.device)
            elif isinstance(batch[1], dict):
                data: dict[str, torch.Tensor] = {}  # type: ignore[no-redef]
                for name, tensor in batch[1].items():
                    data[name] = tensor.to(stateful_model.device)
            else:
                raise ValueError(f"data type {type(batch[1])} not supported")

            with torch.autocast(stateful_model.device_type, enabled=stateful_model.amp):
                stateful_model.model.model(data)
                embeddings = stateful_model.model.model.embedding_recorder.embedding
                if all_embeddings is None:
                    all_embeddings = embeddings
                else:
                    all_embeddings = torch.cat((all_embeddings, embeddings), 0)

    stateful_model.model.model.embedding_recorder.end_recording()
 
    return all_embeddings