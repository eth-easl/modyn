from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration

from modyn.models.coreset_methods_support import CoresetSupportingModule


class T5:
    def __init__(self, model_config: Dict[str, Any], device: str, amp: bool) -> None:
        """
        Args:
            model_config: dict with optional keys
                - "model_name_or_path" (str, default "t5-base")
                - "max_length" (int, default 128)
                - "temperature" (float, default 1.0)
                - "top_k" (int, default 50)
                - "top_p" (float, default 0.95)
                - "num_return_sequences" (int, default 1)
                - "dtype" (torch.dtype, default torch.float32)
            device: e.g. "cuda" or "cpu"
            amp: if True and on CUDA, cast model to half precision
        """
        self.model = T5Modyn(model_config)
        self.model.to(device)


class T5Modyn(CoresetSupportingModule):
    def __init__(self, model_config: Dict[str, Any]) -> None:
        super().__init__()

        # hyperparameters
        self.model_name = model_config.get("model_name_or_path", "t5-base")
        self.max_length = model_config.get("max_length", 128)
        self.temperature = model_config.get("temperature", 1.0)
        self.top_k = model_config.get("top_k", 50)
        self.top_p = model_config.get("top_p", 0.95)
        self.num_return_sequences = model_config.get("num_return_sequences", 1)
        self.dtype = model_config.get("dtype", torch.float32)
         
        # validate model name
        valid = {"t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"}
        assert self.model_name in valid, f"Invalid model name: {self.model_name}"

        # tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=self.dtype
        )

    def forward(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Any:
        """
        If labels is provided: returns the full Seq2SeqLMOutput (with loss & logits).
        If labels is None: returns encoder embeddings (batch, seq_len, d_model).
        """
        input_ids = data[:, :, 0]
        attention_mask = data[:, :, 1]

        if labels is not None:
            if labels.dim() == 3:
                labels = labels[:, :, 0]
            return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits

        # no labels: return encoder hidden states
        encoder_out = self.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        return encoder_out.last_hidden_state

    def get_last_layer(self) -> nn.Module:
        return self.model.lm_head

    def freeze_params(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_params(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generates sequences using stored hyperparameters.
        Args:
            input_ids: (batch, seq_len)
            attention_mask: optional (batch, seq_len)
        Returns:
            generated_ids: (batch * num_return_sequences, max_length)
        """
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            num_return_sequences=self.num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            early_stopping=True,
        )
        return outputs
