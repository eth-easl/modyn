from typing import Any

import torch
from torch import nn
from transformers import AutoTokenizer, GPT2LMHeadModel

from modyn.models.coreset_methods_support import CoresetSupportingModule


class Gpt2:
    def __init__(self, model_config: dict[str, Any], device: str, amp: bool) -> None:
        self.model = Gpt2Modyn(model_config)
        self.model.to(device)


class Gpt2Modyn(CoresetSupportingModule):
    def __init__(self, model_config: dict[str, Any]) -> None:
        super().__init__()

        # Set default values for hyperparameters if not provided in model_config
        model_name = model_config.get("model_name_or_path", "gpt2-large")
        self.model_name = model_name if model_name in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"} else "gpt2-large"
        self.max_length = model_config.get("max_length", 50)
        self.temperature = model_config.get("temperature", 1.0)
        self.top_k = model_config.get("top_k", 50)
        self.top_p = model_config.get("top_p", 0.95)
        self.num_return_sequences = model_config.get("num_return_sequences", 1)
        self.enable_flash_attention = model_config.get("enable_flash_attention", True)
        self.enable_gradient_checkpointing = model_config.get("enable_gradient_checkpointing", False)
        self.dtype = model_config.get("dtype", torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the GPT-2 model with FlashAttention if enabled
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name, torch_dtype=self.dtype)

        if self.enable_flash_attention:
            self.model.config.attn_implementation = "flash_attention"

        # If gradient checkpointing is enabled, enable it
        if self.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, data: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Forward method for text generation or language modeling tasks."""
        input_ids = data[:, :, 0]
        attention_mask = data[:, :, 1]

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def get_last_layer(self) -> nn.Module:
        """Retrieve the last layer (lm_head) of the model."""
        return self.model.lm_head

    def freeze_params(
        self,
    ) -> None:  # We use this when applying extra layers there does not seem to be a more elegant way
        """Freeze model parameters to avoid updates during training."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_params(self) -> None:
        """Unfreeze model parameters for training."""
        for param in self.model.parameters():
            param.requires_grad = True

    def generate(  # We use it for evaluation
        self,
        input_ids: torch.tensor,
    ) -> list:
        """Generate sequences based on the given input tensor."""
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=self.max_length,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            num_return_sequences=self.num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return outputs
