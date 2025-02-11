from typing import Any

import torch
from torch import nn
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, GPT2Model

from modyn.models.coreset_methods_support import CoresetSupportingModule


class Gpt2:
    # pylint: disable-next=unused-argument
    def __init__(self, hparams: Any, device: str, amp: bool) -> None:
        self.model = Gpt2Modyn(hparams)
        self.model.to(device)


"""
Adapted from an example implementation of a GPT-2 model.
This implementation uses the GPT-2 tokenizer from Hugging Face's Transformers library:
https://huggingface.co/docs/transformers/model_doc/gpt2
"""


class Gpt2Modyn(CoresetSupportingModule):
    def __init__(self, hparams: Any) -> None:
        super().__init__()

        # Use hparams to decide the GPT-2 version
        model_name = hparams.model_name_or_path if hasattr(hparams, "model_name_or_path") else "gpt2-large"

        # Assert that the model name is valid
        valid_model_names = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        assert model_name in valid_model_names, f"Invalid model name: {model_name}. Must be one of {valid_model_names}."
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Load the specified GPT-2 model

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.config = GPT2Config.from_pretrained(model_name)
        self.transformer = GPT2Model(self.config)

    def forward(self, data: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Forward method for text generation or language modeling tasks.

        Args:
        - data (torch.Tensor): Tensor of shape (batch_size, seq_len, 2), where
          the last dimension contains token IDs and attention masks.
        - labels (torch.Tensor, optional): Tensor of labels for language modeling tasks.

        Returns:
        - output: The output logits or loss from the GPT-2 model.
        """
        # Split input into token IDs and attention masks
        input_ids = data[:, :, 0]

        attention_mask = data[:, :, 1]
        # Forward pass through GPT-2

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        return output

    def get_last_layer(self) -> nn.Module:
        """Retrieve the last layer (lm_head) of the model.

        Returns:
            The final linear layer of the GPT-2 model.
        """
        return self.model.lm_head

    def freeze_params(self) -> None:
        for par in self.model.parameters():
            par.requires_grad = False
        for par in self.transformer.parameters():
            par.requires_grad = False

    def unfreeze_params(self) -> None:
        for par in self.model.parameters():
            par.requires_grad = True

    def generate(
        self,
        input_ids: torch.tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
    ) -> list:
        # Generate output sequences
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
