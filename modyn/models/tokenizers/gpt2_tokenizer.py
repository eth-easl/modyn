import torch
from transformers import GPT2Tokenizer

from .hf_tokenizer import HFTokenizerTransform


class GPT2TokenizerTransform(HFTokenizerTransform):
    def __init__(self, max_token_length: int = 256, padding_side: str = "right"):
        """Adapted from an example implementation of a GPT-2 tokenizer.

        This implementation uses the GPT-2 tokenizer from Hugging Face's
        Transformers library:
        https://huggingface.co/docs/transformers/model_doc/gpt2
        """
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token to avoid padding errors

        tokenizer.padding_side = padding_side
        super().__init__(tokenizer, max_token_length)

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
        if not sample:  # This is needed here since this tokenizer cannot handle an empty list
            sample = self.tokenizer.eos_token
        tokens = self.tokenizer(
            sample, padding="max_length", truncation=True, max_length=self.max_token_length, return_tensors="pt"
        )
        data = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        data = torch.squeeze(data, dim=0)
        return data
