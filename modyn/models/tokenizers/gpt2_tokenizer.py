from transformers import GPT2Tokenizer

from .hf_tokenizer import HFTokenizerTransform


class GPT2TokenizerTransform(HFTokenizerTransform):
    def __init__(self, max_token_length: int = 64):
        """Adapted from an example implementation of a GPT-2 tokenizer.

        This implementation uses the GPT-2 tokenizer from Hugging Face's
        Transformers library:
        https://huggingface.co/docs/transformers/model_doc/gpt2
        """
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        # tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token to avoid padding errors
        # tokenizer.padding_side = "right"
        tokenizer.add_special_tokens(
            {   "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "mask_token": "<mask>",
            }
        )

        tokenizer.padding_side = "right"
        super().__init__(tokenizer, max_token_length)
