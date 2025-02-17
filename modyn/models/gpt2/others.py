import random
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    RobertaForMaskedLM,
    T5ForConditionalGeneration,
)

from modyn.models.coreset_methods_support import CoresetSupportingModule


class Roberta:
    """Adapted from an example implementation of a RoBERTa model for masked
    language modeling."""

    def __init__(self, hparams: Any, device: str, amp: bool) -> None:
        self.model = RobertaModyn(hparams)
        self.model.to(device)


class RobertaModyn(CoresetSupportingModule):
    def __init__(self, hparams: Any) -> None:
        super().__init__()
        self.model = RobertaForMaskedLM.from_pretrained("roberta-large")
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        self.mask_probability = 0.15
        self.freeze_params()

    def freeze_params(self) -> None:
        for par in self.model.parameters():
            par.requires_grad = False

    def unfreeze_params(self) -> None:
        for par in self.model.parameters():
            par.requires_grad = True

    def mask_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        masked_input_ids = input_ids.clone()
        for i in range(masked_input_ids.size(1)):
            if random.random() < self.mask_probability and masked_input_ids[0, i] not in [
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
            ]:
                masked_input_ids[0, i] = self.tokenizer.mask_token_id
        return masked_input_ids

    def forward(self, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        input_ids = data[:, :, 0]
        attention_mask = data[:, :, 1]
        masked_input_ids = self.mask_input(input_ids)
        output = self.model(input_ids=masked_input_ids, attention_mask=attention_mask, labels=labels)
        return output.logits


class T5:
    """Adapted from an example implementation of a T5 model for sequence-to-
    sequence tasks."""

    def __init__(self, hparams: Any, device: str, amp: bool) -> None:
        self.model = T5Modyn(hparams)
        self.model.to(device)


class T5Modyn(CoresetSupportingModule):
    def __init__(self, hparams: Any) -> None:
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-large")
        self.tokenizer = AutoTokenizer.from_pretrained("t5-large")
        self.freeze_params()

    def freeze_params(self) -> None:
        for par in self.model.parameters():
            par.requires_grad = False

    def unfreeze_params(self) -> None:
        for par in self.model.parameters():
            par.requires_grad = True

    def forward(self, data: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        input_ids = data[:, :, 0]
        attention_mask = data[:, :, 1]
        decoder_input_ids = labels if labels is not None else torch.full_like(input_ids, self.tokenizer.pad_token_id)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        return output.logits

    def generate(self, prompt: str, max_length: int = 50) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        output = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class LLaMA:
    """Adapted from an example implementation of a LLaMA model for causal
    language modeling."""

    def __init__(self, hparams: Any, device: str, amp: bool) -> None:
        self.model = LLaMAModyn(hparams)
        self.model.to(device)


class LLaMAModyn(CoresetSupportingModule):
    def __init__(self, hparams: Any) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-1B")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-1B")
        self.freeze_params()

    def freeze_params(self) -> None:
        for par in self.model.parameters():
            par.requires_grad = False

    def unfreeze_params(self) -> None:
        for par in self.model.parameters():
            par.requires_grad = True

    def forward(self, data: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        input_ids = data[:, :, 0]
        attention_mask = data[:, :, 1]
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.logits

    def generate(self, prompt: str, max_length: int = 50) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        output = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
