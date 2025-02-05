import torch
from torch import nn


# LoRA Layer (using your provided default parameters)
class LoRALayer(nn.Module):
    def __init__(self, n_in: int, n_out: int | None = None, adapter_dim: int = 16, adapter_alpha: int = 32):
        super().__init__()
        if not n_out:
            n_out = n_in
        self.adapter_dim = adapter_dim
        self.adapter_alpha = adapter_alpha
        self.adapter_proj_1 = nn.Linear(n_in, adapter_dim, bias=False)
        nn.init.normal_(self.adapter_proj_1.weight, std=0.02)
        self.adapter_proj_2 = nn.Linear(adapter_dim, n_out, bias=False)
        self.adapter_proj_2.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_factor = self.adapter_dim / self.adapter_alpha
        result = torch.matmul(x, self.adapter_proj_1.weight.type_as(x).T)
        return torch.matmul(result, self.adapter_proj_2.weight.type_as(x).T) * scale_factor


# KAdapter Layer
class KAdapter(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)  # Residual connection


# Function to apply LoRA
def apply_lora(
    model: nn.Module, target_modules: list[str] | None = None, adapter_dim: int = 16, adapter_alpha: int = 32
) -> nn.Module:
    if target_modules is None:
        target_modules = ["query", "value"]

    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            n_in = module.out_features if hasattr(module, "out_features") else module.weight.shape[0]
            setattr(model, f"lora_{name}", LoRALayer(n_in, adapter_dim=adapter_dim, adapter_alpha=adapter_alpha))
    return model


# Function to apply KAdapter
def apply_kadapter(model: nn.Module) -> nn.Module:
    for _, module in model.named_modules():
        if hasattr(module, "output") and hasattr(module.output, "dense"):  # Transformer layers
            hidden_dim = module.output.dense.out_features
            module.kadapter = KAdapter(hidden_dim)
    return model
