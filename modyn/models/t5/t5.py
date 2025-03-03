from typing import Any

import torch
from torch import nn
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration

from modyn.models.coreset_methods_support import CoresetSupportingModule


class T5:
    # pylint: disable-next=unused-argument
    def __init__(self, hparams: Any, device: str, amp: bool) -> None:
        """
        Initializes the T5 model and sends it to the specified device.

        Args:
            hparams (Any): Hyperparameters for the model.
            device (str): Device to use ('cuda' or 'cpu').
            amp (bool): Whether to use automatic mixed precision.
        """
        self.model = T5Modyn(hparams)
        self.model.to(device)

class T5Modyn(CoresetSupportingModule):
    def __init__(self, hparams: Any) -> None:
        super().__init__()

        # Use hparams to decide the T5 version
        model_name = hparams.model_name_or_path if hasattr(hparams, "model_name_or_path") else "t5-base"

        # Assert that the model name is valid
        valid_model_names = {"t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"}
        assert model_name in valid_model_names, f"Invalid model name: {model_name}. Must be one of {valid_model_names}."

        # Load the T5 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the specified T5 model
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.config = T5Config.from_pretrained(model_name)

    def forward(self, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for T5.

        Args:
            data (torch.Tensor): Tensor of shape (batch_size, seq_len, 2), where
                the last dimension contains token IDs and attention masks.
            labels (torch.Tensor, optional): Tensor of labels for sequence-to-sequence tasks.

        Returns:
            output: The output logits or loss from the T5 model.
        """
        # Split input into token IDs and attention masks
        input_ids = data[:, :, 0]
        attention_mask = data[:, :, 1]
        print(input_ids.shape, attention_mask.shape, labels.shape)
        labels = labels[:, :, 0]
        # Forward pass through T5
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        return output

    def get_last_layer(self) -> nn.Module:
        """
        Retrieve the last layer (lm_head) of the model.

        Returns:
            The final linear layer of the T5 model.
        """
        return self.model.lm_head

    def freeze_params(self) -> None:
        """Freezes all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_params(self) -> None:
        """Unfreezes all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def generate(
        self,
        input_texts: list,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
    ) -> list:
        """
        Generates text sequences from input texts.

        Args:
            input_texts (list): List of input strings.
            max_length (int): Maximum length of generated sequence.
            temperature (float): Sampling temperature.
            top_k (int): Top-k filtering.
            top_p (float): Top-p (nucleus) sampling.
            num_return_sequences (int): Number of sequences to generate.

        Returns:
            list: List of generated text sequences.
        """
        input_encodings = self.tokenizer(
            input_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )

        # Generate output sequences
        outputs = self.model.generate(
            input_ids=input_encodings["input_ids"],
            attention_mask=input_encodings["attention_mask"],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
