import torch
from torch.utils.data import DataLoader
from transformers import T5Config
from modyn.supervisor.internal.triggers.utils.model.stateful_model import StatefulModel

def get_embeddings(stateful_model: StatefulModel, dataloader: DataLoader) -> torch.Tensor:
    """
    input: embedding_encoder with downloaded model (T5Modyn or other)
    output: embeddings Tensor
    """
    assert stateful_model.model is not None
    wrapped = stateful_model.model.model
    wrapped.eval()

    # detect T5
    from transformers import T5Config
    cfg = getattr(wrapped, "config", None)
    is_t5 = isinstance(cfg, T5Config) and cfg.model_type == "t5"
    if is_t5:
        wrapped.model.shared.register_forward_hook(
            lambda m, i, o: stateful_model.model.model.embedding_recorder.record(o)
        )

    stateful_model.model.model.embedding_recorder.start_recording()
    all_embeddings = None

    with torch.no_grad():
        for _, batch in dataloader:
            raw = batch[1]
            if is_t5:
                # unpack tensor→ids+mask or dict→ids+mask
                if isinstance(raw, torch.Tensor):
                    t = raw.to(stateful_model.device)
                    input_ids, attention_mask = t[..., 0], t[..., 1]
                elif isinstance(raw, dict):
                    input_ids = raw["input_ids"].to(stateful_model.device)
                    attention_mask = raw["attention_mask"].to(stateful_model.device)
                else:
                    raise ValueError(f"{type(raw)} not supported for T5")

                with torch.autocast(stateful_model.device_type, enabled=stateful_model.amp):
                    _ = wrapped.model.encoder(input_ids=input_ids, attention_mask=attention_mask)

            else:
                # original path
                if isinstance(raw, torch.Tensor):
                    data = raw.to(stateful_model.device)
                elif isinstance(raw, dict):
                    data = {k: v.to(stateful_model.device) for k, v in raw.items()}
                else:
                    raise ValueError(f"{type(raw)} not supported")

                with torch.autocast(stateful_model.device_type, enabled=stateful_model.amp):
                    _ = wrapped(data)

            emb = stateful_model.model.model.embedding_recorder.embedding
            all_embeddings = emb if all_embeddings is None else torch.cat((all_embeddings, emb), dim=0)

    stateful_model.model.model.embedding_recorder.end_recording()
    return all_embeddings
