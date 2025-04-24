import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from modyn.models import (
    apply_adapters,
    apply_kadapter,
    apply_peft_adapter,
)
from modyn.utils.utils import count_trainable_params


# --- Our MockTransformer Model ---
class MockTransformer:
    def __init__(self, hparams, device, amp):
        self.hparams = hparams
        self.device = device
        self.amp = amp
        # Load the pretrained model.
        base_model = AutoModelForCausalLM.from_pretrained(hparams.model_name_or_path)
        # Wrap the model in our ModelWrapper.
        self.model = ModelWrapper(base_model)


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        # Save the underlying model and expose its config.
        self.model = model
        self.config = model.config

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def freeze_params(self):
        for param in self.model.parameters():
            param.requires_grad = False


class HParams:
    def __init__(self, model_name_or_path="gpt2-large", device="cpu", amp=False):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.amp = amp


def test_apply_kadapter():
    hparams = HParams()
    model = MockTransformer(hparams, hparams.device, hparams.amp)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token

    # Apply Kadapter (which attaches the adapter to the inner model).
    modified_model = apply_kadapter(model.model)

    if not hasattr(modified_model, "kadapter"):
        assert any(
            "kadapter" in name for name, _ in modified_model.named_parameters()
        ), "KAdapter not attached to the inner model."


# Test for training with Kadapters
def test_model_training_with_kadapters():
    hparams = HParams()
    model = MockTransformer(hparams, hparams.device, hparams.amp)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token

    model.model = apply_kadapter(model.model)
    input_text = "Once upon a time, a king had a dream."
    encoding = tokenizer(input_text, return_tensors="pt", padding=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    labels = input_ids.clone()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.model.parameters()), lr=5e-5)
    model.model.train()

    with torch.no_grad():
        initial_outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
    initial_logits = initial_outputs.logits.clone().detach()

    for _ in range(2):
        optimizer.zero_grad()
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
    final_logits = final_outputs.logits.clone().detach()

    assert loss.item() > 0, "Loss should be > 0 after training."
    assert not torch.equal(initial_logits, final_logits), "Logits should change after training."

    # Verify that the adapter parameters have received gradients.
    for name, param in model.model.named_parameters():
        if "kadapter" in name:
            assert param.grad is not None, f"Expected gradient for {name} but found None."
        else:
            assert param.grad is None or torch.all(param.grad == 0), f"Unexpected gradient in {name}."


def test_apply_lora():
    hparams = HParams()
    model = MockTransformer(hparams, hparams.device, hparams.amp)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token

    model.model = apply_peft_adapter(
        model.model, "lora", target_modules=["c_attn", "c_proj"], r=16, lora_alpha=32
    ).model

    for name, param in model.model.model.named_parameters():
        if "lora" in name:
            assert param.requires_grad, f"LoRA parameter {name} should be trainable."
        else:
            assert not param.requires_grad, f"Non-LoRA parameter {name} should be frozen."


def test_apply_prompt_tuning():
    hparams = HParams()
    model = MockTransformer(hparams, hparams.device, hparams.amp)
    params_before = count_trainable_params(model.model)
    model.model = apply_peft_adapter(model.model, "prompt_tuning", num_virtual_tokens=10).model
    params_after = count_trainable_params(model.model)
    assert params_after != params_before, "Prompt tuning did not modify trainable parameters."


# Updated test for Prefix Tuning
def test_apply_prefix_tuning():
    hparams = HParams()
    model = MockTransformer(hparams, hparams.device, hparams.amp)
    params_before = count_trainable_params(model.model)
    model.model = apply_peft_adapter(model.model, "prefix_tuning", num_virtual_tokens=15).model
    params_after = count_trainable_params(model.model)
    assert params_after != params_before, "Prefix tuning did not modify trainable parameters."


def test_model_with_adapters_inference():
    hparams = HParams()
    model = MockTransformer(hparams, hparams.device, hparams.amp)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token

    model.model = apply_peft_adapter(model.model, "lora", target_modules=["c_attn", "c_proj"]).model
    model.model = apply_kadapter(model.model).model

    input_text = "Hello, world!"
    encoding = tokenizer(input_text, return_tensors="pt", padding=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    model.model.eval()
    with torch.no_grad():
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    vocab_size = model.model.get_input_embeddings().weight.shape[0]
    assert logits.shape[-1] == vocab_size, "Output dimension mismatch."


def test_model_training_with_lora():
    hparams = HParams()
    model = MockTransformer(hparams, hparams.device, hparams.amp)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token

    model.model = apply_peft_adapter(
        model.model, "lora", target_modules=["c_attn", "c_proj"], r=16, lora_alpha=32
    ).model

    input_text = "Once upon a time, a king had a dream."
    encoding = tokenizer(input_text, return_tensors="pt", padding=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    labels = input_ids.clone()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.model.parameters()), lr=5e-5)
    model.model.train()

    with torch.no_grad():
        initial_outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
    initial_logits = initial_outputs.logits.clone().detach()

    for _ in range(2):
        optimizer.zero_grad()
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
    final_logits = final_outputs.logits.clone().detach()

    assert loss.item() > 0, "Loss should be > 0 after training."
    assert not torch.equal(initial_logits, final_logits), "Logits should change after training."

    for name, param in model.model.named_parameters():
        if "lora" in name:
            assert param.grad is not None, f"Expected gradient for {name} but found None."
        else:
            assert param.grad is None or torch.all(param.grad == 0), f"Unexpected gradient in {name}."


# Test for applying adapters with default arguments
def test_apply_adapters_default_args():
    hparams = HParams()
    model = MockTransformer(hparams, hparams.device, hparams.amp)

    model.model = apply_adapters(model.model, ["lora", "kadapter"], {}).model

    lora_params = [name for name, _ in model.model.model.named_parameters() if "lora" in name]
    assert len(lora_params) > 0, "LoRA adapter was not applied with default args."
    # Check that Kadapter exists on the inner model (directly or via parameter names).
    assert hasattr(model.model, "kadapter") or any(
        "kadapter" in name for name, _ in model.model.named_parameters()
    ), "KAdapter adapter was not applied with default args."


# Test for applying adapters with explicit arguments
def test_apply_adapters_with_args():
    hparams = HParams()
    model = MockTransformer(hparams, hparams.device, hparams.amp)

    adapter_args = {
        "lora": {"r": 4, "lora_alpha": 8},
        "kadapter": {"adapter_layers": [2], "scale_factor": 0.05},
    }
    model.model = apply_adapters(
        model.model, ["lora", "kadapter", "prefix_tuning", "prompt_tuning"], adapter_args
    ).model

    lora_params = [name for name, _ in model.model.model.named_parameters() if "lora" in name]
    assert len(lora_params) > 0, "LoRA adapter was not applied with args."
    assert hasattr(model.model, "kadapter") or any(
        "kadapter" in name for name, _ in model.model.named_parameters()
    ), "KAdapter adapter was not applied with args."


# Test that applying no adapters leaves the model unchanged.
def test_apply_adapters_empty_list():
    hparams = HParams()
    model = MockTransformer(hparams, hparams.device, hparams.amp)
    original_state = dict(model.model.named_parameters())
    model.model = apply_adapters(model.model, [], {})
    final_state = dict(model.model.named_parameters())
    assert original_state.keys() == final_state.keys(), "Model structure changed despite no adapters."
