# hf_tokenizer.py

import torch
from transformers import PreTrainedTokenizer


class HFTokenizerTransform:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_token_length: int) -> None:
        """
        Parent class for tokenizers based on HuggingFace's Transformers.
        Args:
            tokenizer: Preloaded tokenizer object.
            max_token_length: Maximum length for tokenization.
        """
        self.max_token_length = max_token_length
        self.tokenizer = tokenizer

    def __call__(self, sample: str) -> torch.Tensor:
        """
        Tokenize the input sample and return a tensor with input_ids and attention_mask.
        Args:
            sample: Input string to tokenize.
        Returns:
            A torch.Tensor with shape (max_token_length, 2), where:
            - dim 0 is the token sequence length.
            - dim 1 contains input_ids and attention_mask.
        """
        tokens = self.tokenizer(
            sample, padding="max_length", truncation=True, max_length=self.max_token_length, return_tensors="pt"
        )
        data = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        data = torch.squeeze(data, dim=0)
        return data
