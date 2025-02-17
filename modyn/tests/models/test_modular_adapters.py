import torch
from transformers import AutoTokenizer

from modyn.models import Gpt2, apply_lora

# =============================================================================
# Updated Test Suite
# =============================================================================

# Note: The tests below assume that you're using a GPT-2 model from transformers.
# If you want to use your custom Gpt2 class from modyn.models, adjust the imports accordingly.


class HParams:
    def __init__(self, model_name_or_path="gpt2-large", device="cpu", amp=False):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.amp = amp


def test_apply_lora():
    hparams = HParams()
    model = Gpt2(hparams, hparams.device, hparams.amp)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    target_modules = ["c_attn", "c_proj"]
    model.model = apply_lora(model.model, target_modules=target_modules, adapter_dim=16, adapter_alpha=32)
    # Verify that LoRA parameters are trainable and others are frozen.

    for name, param in model.model.model.named_parameters():
        if "lora" in name:
            assert param.requires_grad, f"LoRA parameter {name} should be trainable."
        else:
            assert not param.requires_grad, f"Non-LoRA parameter {name} should be frozen."


def test_model_with_adapters_inference():
    hparams = HParams()
    model = Gpt2(hparams, hparams.device, hparams.amp)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    model.model = apply_lora(model.model, target_modules=["c_attn", "c_proj"])

    input_text = "Hello, world!"
    encoding = tokenizer(input_text, return_tensors="pt", padding=True)
    input_ids = encoding["input_ids"]  # shape: (1, seq_len)
    attention_mask = encoding["attention_mask"]  # shape: (1, seq_len)
    data = torch.stack([input_ids, attention_mask], dim=-1)
    model.model.eval()
    with torch.no_grad():
        outputs = model.model(data)
        logits = outputs.logits
    print(logits.shape[2])
    shape = model.model.model.get_input_embeddings().weight.shape[0]

    assert logits.shape[2] == shape, "Output dimension mismatch."




def test_model_training_with_lora():
    hparams = HParams()
    model = Gpt2(hparams, hparams.device, hparams.amp)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    target_modules = ["c_attn", "c_proj"]
    model = apply_lora(model.model, target_modules=target_modules, adapter_dim=16, adapter_alpha=32)

    input_text = "Once upon a time, a king had a dream."
    encoding = tokenizer(input_text, return_tensors="pt", padding=True)
    input_ids = encoding["input_ids"]  # shape: (1, seq_len)
    attention_mask = encoding["attention_mask"]  # shape: (1, seq_len)
    data = torch.stack([input_ids, attention_mask], dim=-1)
    labels = input_ids.clone()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    model.train()

    # Capture initial output
    with torch.no_grad():
        initial_outputs = model(data)
    initial_logits = initial_outputs.logits.clone().detach()

    # Small training loop
    for _ in range(2):
        optimizer.zero_grad()
        outputs = model(data, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Capture final output
    with torch.no_grad():
        final_outputs = model(data)
    final_logits = final_outputs.logits.clone().detach()

    # Check that loss is > 0 and that the model's output changes
    assert loss.item() > 0, "Loss should be > 0 after training."
    assert not torch.equal(initial_logits, final_logits), "Logits should change after training."

    # Verify that gradients exist only for LoRA parameters
    for name, param in model.named_parameters():
        if "lora" in name:
            assert param.grad is not None, f"Expected gradient for {name} but found None."
        else:
            assert param.grad is None or torch.all(param.grad == 0), f"Unexpected gradient in {name}."

