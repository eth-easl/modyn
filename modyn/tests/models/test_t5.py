import torch
from transformers import AutoTokenizer

from modyn.models import T5


class HParams(dict):
    def __init__(self):
        super().__init__({
            "model_name_or_path": "t5-base",
            "max_length": 128,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
            "num_return_sequences": 1,
            "dtype": torch.float32
        })
        self.device = "cpu"
        self.amp = False



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

    input_texts = [
        "translate English to French: I love programming",
        "translate English to French: How are you today?",
        "translate English to French: This is a test sentence.",
        "translate English to French: The weather is nice."
    ]

    tokens = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)

    # Stack input_ids and attention_mask into final shape (B, L, 2)
    stacked_input = torch.stack([tokens.input_ids, tokens.attention_mask], dim=-1)

    print(f"Stacked input shape: {stacked_input.shape}")  # Should be (4, seq_len, 2)

    # Use your custom generate() wrapper
    generated = model.model.generate(stacked_input)
    print(f"Type of outputs: {type(generated)}")
    print(f"Generated token IDs shape: {generated.shape}")
    print(f"Generated token IDs:\n{generated}")
    print(generated.shape)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    print("Decoded outputs:")
    for i, text in enumerate(decoded):
        print(f"[{i}] {text}")
    
    assert isinstance(decoded, list)
    assert len(decoded) == len(input_texts)
    for output in decoded:
        assert isinstance(output, str)
        assert len(output) > 0


test_text_generation()