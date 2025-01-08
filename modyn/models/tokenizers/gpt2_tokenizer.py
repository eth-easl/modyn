# gpt2_tokenizer.py

from transformers import GPT2Tokenizer

from .hf_tokenizer import HFTokenizerTransform


class GPT2TokenizerTransform(HFTokenizerTransform):
    def __init__(self, max_token_length: int = 256):
        """
        Transformer for GPT-2.
        Args:
            max_token_length: Maximum token length for GPT-2.
        """
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token to avoid padding errors
        tokenizer.padding_side = "right"
        super().__init__(tokenizer, max_token_length)
