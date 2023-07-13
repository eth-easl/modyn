import torch
from transformers import DistilBertTokenizer


class DistilBertTokenizerTransform:
    """
    Adapted from WildTime's initialize_distilbert_transform
    Here you can find the original implementation:
    https://github.com/huaxiuyao/Wild-Time/blob/main/wildtime/data/utils.py
    """

    def __init__(self, max_token_length: int = 300) -> None:
        self.max_token_length = max_token_length
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def __call__(self, sample: str) -> torch.Tensor:
        tokens = self.tokenizer(
            sample, padding="max_length", truncation=True, max_length=self.max_token_length, return_tensors="pt"
        )
        data = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        data = torch.squeeze(data, dim=0)  # First shape dim is always 1
        return data
