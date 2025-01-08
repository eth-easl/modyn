# distill_bert_tokenizer.py

from transformers import DistilBertTokenizer

from .hf_tokenizer import HFTokenizerTransform


class DistilBertTokenizerTransform(HFTokenizerTransform):
    def __init__(self, max_token_length: int = 300):
        """
        Transformer for DistilBERT.
        Args:
            max_token_length: Maximum token length for DistilBERT.
        """
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        super().__init__(tokenizer, max_token_length)
