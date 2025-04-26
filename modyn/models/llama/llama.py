from typing import Any

import torch
from torch import nn
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from modyn.models.coreset_methods_support import CoresetSupportingModule


class Llama:
    def __init__(self, hparams: Any, device: str, amp: bool) -> None:
        self.model = LlamaModyn(hparams)
        self.model.to(device)


class LlamaModyn(CoresetSupportingModule):
    def __init__(self, hparams: Any) -> None:
        super().__init__()

        # Set defaults
        model_name = getattr(hparams, "model_name_or_path", "meta-llama/Llama-2-7b-hf")
        self.model_name = model_name
        self.max_length = getattr(hparams, "max_length", 50)
        self.temperature = getattr(hparams, "temperature", 1.0)
        self.top_k = getattr(hparams, "top_k", 50)
        self.top_p = getattr(hparams, "top_p", 0.95)
        self.num_return_sequences = getattr(hparams, "num_return_sequences", 1)
        self.enable_flash_attention = getattr(hparams, "enable_flash_attention", True)
        self.enable_gradient_checkpointing = getattr(hparams, "enable_gradient_checkpointing", False)
        self.dtype = getattr(hparams, "dtype", torch.float32)

        # Load tokenizer (LLaMA uses SentencePiece-based tokenizers)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # LLaMA may not have pad_token set
        self.tokenizer.padding_side = "left"

        # Load model
        self.model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)

        if self.enable_flash_attention:
            self.model.config.attn_implementation = "flash_attention"

        if self.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.config = LlamaConfig.from_pretrained(self.model_name)

    def forward(self, data: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        input_ids = data[:, :, 0]
        attention_mask = data[:, :, 1]
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def get_last_layer(self) -> nn.Module:
        return self.model.lm_head

    def freeze_params(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_params(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=self.max_length,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            num_return_sequences=self.num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return outputs
