from peft import LoraConfig, TaskType, get_peft_model
from torch import nn


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_lora(
    model: nn.Module,
    target_modules: list[str] | None = ["c_attn", "c_proj"],
    adapter_dim: int = 16,
    adapter_alpha: int = 32,
) -> nn.Module:
    # Count and print trainable parameters before applying LoRA
    trainable_params_before = count_trainable_params(model)
    print(f"\n Trainable parameters BEFORE applying LoRA: {trainable_params_before}")

    # Create a LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=adapter_dim,
        lora_alpha=adapter_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
    )

    # Wrap the model with LoRA
    model.model = get_peft_model(model.model, lora_config)

    # Count and print trainable parameters after applying LoRA
    trainable_params_after = count_trainable_params(model)
    print(f"\n Trainable parameters AFTER applying LoRA: {trainable_params_after}")

    return model
