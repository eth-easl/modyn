from transformers import DistilBertTokenizer

from .hf_tokenizer import HFTokenizerTransform


class DistilBertTokenizerTransform(HFTokenizerTransform):
    def __init__(self, max_token_length: int = 300):
        """
        Adapted from WildTime's initialize_distilbert_transform
        Here you can find the original implementation:
        https://github.com/huaxiuyao/Wild-Time/blob/main/wildtime/data/utils.py
        """
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        super().__init__(tokenizer, max_token_length)
