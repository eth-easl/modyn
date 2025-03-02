from transformers import T5Tokenizer
from .hf_tokenizer import HFTokenizerTransform


class T5TokenizerTransform(HFTokenizerTransform):
    def __init__(self, max_token_length: int = 64):
        """
        Tokenizer transform class for the T5 model.

        Args:
            max_token_length: Maximum length for tokenization (default: 300).
        """
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        super().__init__(tokenizer, max_token_length)
