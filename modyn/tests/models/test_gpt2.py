import torch
from transformers import AutoTokenizer

from modyn.models import Gpt2


class HParams:
    def __init__(self, model_name_or_path="gpt2-large", device="cpu", amp=False):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.amp = amp


def test_gpt2modyn_initialization():
    hparams = HParams()
    model = Gpt2(hparams, hparams.device, hparams.amp)  # Pass device and amp explicitly
    assert isinstance(model.model, torch.nn.Module)


def test_forward_pass():
    hparams = HParams()
    model = Gpt2(hparams, hparams.device, hparams.amp)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    text = ["Hello, how are you?"]
    tokens = tokenizer(text, return_tensors="pt", padding=True)

    input_data = torch.stack([tokens.input_ids, tokens.attention_mask], dim=-1)
    output = model.model(input_data)  # Fix incorrect model call

    assert output.logits.shape[-1] == tokenizer.vocab_size  # Logits over vocab size


def test_get_last_layer():
    hparams = HParams()
    model = Gpt2(hparams, hparams.device, hparams.amp)
    last_layer = model.model.get_last_layer()  # Ensure method is correctly accessed

    assert isinstance(last_layer, torch.nn.Linear)
    assert last_layer.out_features == 50257


def test_freeze_unfreeze_params():
    hparams = HParams()
    model = Gpt2(hparams, hparams.device, hparams.amp)

    model.model.freeze_params()
    assert all(not param.requires_grad for param in model.model.parameters())
    model.model.unfreeze_params()
    assert all(param.requires_grad for param in model.model.parameters())


def test_text_generation():
    hparams = HParams()
    model = Gpt2(hparams, hparams.device, hparams.amp)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

    input_text = "Once upon a time"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    generated_texts = model.model.generate(input_ids, max_length=20, num_return_sequences=1)

    assert isinstance(generated_texts, list)
    assert len(generated_texts) == 1
    assert isinstance(generated_texts[0], str)