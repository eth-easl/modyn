import torch
from transformers import GPT2Tokenizer


class GPT2TokenizerTransform:
    def __init__(self, max_token_length: int = 256):
        # Load the GPT-2 tokenizer
        self.max_token_length = max_token_length
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        # Set the pad token to the eos token to avoid padding errors
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def __call__(self, sample: str) -> torch.Tensor:
        # Make the class callable to use it as Torch Transform
        tokens = self.tokenizer(
            sample, padding="max_length", truncation=True, max_length=self.max_token_length, return_tensors="pt"
        )
        # Create a tensor whose first dimension is the input_ids and the second is the attention_mask
        data = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        data = torch.squeeze(data, dim=0)  # First shape dim is always 1, since the input is just one string
        return data
