from transformers import LlamaTokenizer

from .hf_tokenizer import HFTokenizerTransform


class LlamaTokenizerTransform(HFTokenizerTransform):
    def __init__(self, max_token_length: int = 512):
        """
        Tokenizer transform class for the LLaMA model.

        Args:
            max_token_length: Maximum length for tokenization (default: 512).
        """
        tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        super().__init__(tokenizer, max_token_length)
