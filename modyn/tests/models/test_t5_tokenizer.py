import torch

from modyn.models.tokenizers import T5TokenizerTransform


def test_t5_tokenizer_transform():
    max_token_length = 40
    transform = T5TokenizerTransform(max_token_length=max_token_length)

    # Non-empty input
    input_text = "This is a test sentence."

    output = transform(input_text)

    # Check shape and dtype
    assert output.shape == (max_token_length, 2)
    assert output.dtype == torch.int64

    input_ids = output[:, 0]
    attention_mask = output[:, 1]

    # T5 typically uses <pad> = 0, so we expect the first tokens to be non-zero
    # and eventually trailing zeros once the text ends.
    assert torch.sum(input_ids[:5]) != 0  # the first few tokens are actual words
    # Verify where input_ids != 0, the mask is 1
    for i, token_id in enumerate(input_ids):
        if token_id != 0:
            assert attention_mask[i] == 1


def test_t5_tokenizer_transform_empty_input():
    max_token_length = 40
    transform = T5TokenizerTransform(max_token_length=max_token_length)

    # Empty input
    input_text = ""

    output = transform(input_text)

    # Check shape/dtype
    assert output.shape == (max_token_length, 2)
    assert output.dtype == torch.int64

    input_ids = output[:, 0]
    attention_mask = output[:, 1]

    assert attention_mask[0] == 1

    assert torch.sum(input_ids) == 1
    assert torch.sum(attention_mask) == 1
