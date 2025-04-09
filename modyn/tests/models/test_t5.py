import torch
from transformers import AutoTokenizer

from modyn.models import T5


class HParams:
    def __init__(self, model_name_or_path="t5-base", device="cpu", amp=False):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.amp = amp


def test_t5modyn_initialization():
    hparams = HParams()
    model = T5(hparams, hparams.device, hparams.amp)
    assert isinstance(model.model.model, torch.nn.Module)  # Inner .model.model is the HF model


def test_forward_pass():
    hparams = HParams()
    model = T5(hparams, hparams.device, hparams.amp)
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(["translate English to German: Hello, how are you?"], padding=True, return_tensors="pt")
    labels = tokenizer(["Hallo, wie geht es dir?"], padding=True, return_tensors="pt").input_ids

    # Create stacked input with input_ids and attention_mask
    input_data = torch.stack([inputs.input_ids, inputs.attention_mask], dim=-1)
    label_data = torch.stack([labels, torch.ones_like(labels)], dim=-1)  # Add dummy attention for compatibility

    output = model.model(input_data, label_data)

    assert output.shape[-1] == model.model.config.vocab_size


def test_get_last_layer():
    hparams = HParams()
    model = T5(hparams, hparams.device, hparams.amp)
    last_layer = model.model.get_last_layer()

    assert isinstance(last_layer, torch.nn.Linear)
    assert last_layer.out_features == model.model.config.vocab_size


def test_freeze_unfreeze_params():
    hparams = HParams()
    model = T5(hparams, hparams.device, hparams.amp)

    model.model.freeze_params()
    assert all(not p.requires_grad for p in model.model.parameters())

    model.model.unfreeze_params()
    assert all(p.requires_grad for p in model.model.parameters())


def test_text_generation():
    hparams = HParams()
    model = T5(hparams, hparams.device, hparams.amp)
    tokenizer = model.model.tokenizer

    input_text = "translate English to French: I love programming"
    tokens = tokenizer(input_text, return_tensors="pt", padding=True)
    stacked_input = torch.stack([tokens.input_ids, tokens.attention_mask], dim=-1)

    generated = model.model.generate(stacked_input, max_length=20, num_return_sequences=1)

    # Decode generated tokens
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

    assert isinstance(decoded, list)
    assert len(decoded) == 1
    assert isinstance(decoded[0], str)
    assert len(decoded[0]) > 0
