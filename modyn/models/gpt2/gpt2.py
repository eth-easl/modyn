from typing import Any

import torch
from torch import nn
from transformers import GPT2LMHeadModel

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

        self.model = GPT2LMHeadModel.from_pretrained("gpt2-large")  # hparams.model_name_or_path

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
        return output[0]

    def get_last_layer(self) -> nn.Module:
        """Retrieve the last layer (lm_head) of the model.

        Returns:
            The final linear layer of the GPT-2 model.
        """
        return self.model.lm_head
