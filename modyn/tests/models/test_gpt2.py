import torch
from transformers import AutoTokenizer

from modyn.models import Gpt2


def test_gpt2modyn_initialization():
    model_config = {"model_name_or_path": "gpt2-large"}
    device = "cpu"
    amp = False
    model = Gpt2(model_config, device, amp)
    assert isinstance(model.model, torch.nn.Module)


def test_forward_pass():
    model_config = {"model_name_or_path": "gpt2-large"}
    device = "cpu"
    amp = False
    model = Gpt2(model_config, device, amp)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    text = ["Hello, how are you?"]
    tokens = tokenizer(text, return_tensors="pt", padding=True)

    input_data = torch.stack([tokens.input_ids, tokens.attention_mask], dim=-1)
    output = model.model(input_data)

    assert output.logits.shape[-1] == tokenizer.vocab_size  # Logits over vocab size


def test_get_last_layer():
    model_config = {"model_name_or_path": "gpt2-large"}
    device = "cpu"
    amp = False
    model = Gpt2(model_config, device, amp)
    last_layer = model.model.get_last_layer()

    assert isinstance(last_layer, torch.nn.Linear)
    assert last_layer.out_features == 50257


def test_freeze_unfreeze_params():
    model_config = {"model_name_or_path": "gpt2-large"}
    device = "cpu"
    amp = False
    model = Gpt2(model_config, device, amp)

    model.model.freeze_params()
    assert all(not param.requires_grad for param in model.model.parameters())
    model.model.unfreeze_params()
    assert all(param.requires_grad for param in model.model.parameters())


def test_text_generation():
    model_config = {"model_name_or_path": "gpt2-large"}
    device = "cpu"
    amp = False
    model = Gpt2(model_config, device, amp)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

    input_text = "Once upon a time"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    generated_texts = model.model.generate(input_ids)

    assert isinstance(generated_texts, torch.Tensor)
    assert len(generated_texts) == 1
    assert isinstance(generated_texts[0], torch.Tensor)


def test_gradient_checkpointing():
    model_config = {
        "enable_flash_attention": True,
        "enable_gradient_checkpointing": True,
        "model_name_or_path": "gpt2-large",
    }
    model = Gpt2(model_config, "cpu", False)

    # Each transformer block should have gradient checkpointing enabled
    assert all(getattr(block, "gradient_checkpointing", True) for block in model.model.model.transformer.h)
