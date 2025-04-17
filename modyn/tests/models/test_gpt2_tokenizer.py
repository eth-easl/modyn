import torch

from modyn.models.tokenizers import GPT2TokenizerTransform


def test_gpt2_tokenizer_transform():
    max_token_length = 40
    transform = GPT2TokenizerTransform(max_token_length=max_token_length)

    # Non-empty input string
    input_text = "This is a test sentence."

    output = transform(input_text)

    # Check the shape and dtype
    assert output.shape == (max_token_length, 2)
    assert output.dtype == torch.int64

    input_ids = output[:, 0]
    attention_mask = output[:, 1]

    assert torch.sum(input_ids[:5]) != 0

    for i in range(len(input_ids)):
        if input_ids[i] != transform.tokenizer.pad_token_id:
            assert attention_mask[i] == 1


def test_gpt2_tokenizer_transform_empty_input():
    max_token_length = 40
    transform = GPT2TokenizerTransform(max_token_length=max_token_length)

    # Empty input string
    input_text = ""

    output = transform(input_text)
    print(output)

    assert output.shape == (max_token_length, 2)
    assert output.dtype == torch.int64

    input_ids = output[:, 0]
    attention_mask = output[:, 1]

    assert input_ids[0] == transform.tokenizer.pad_token_id

    assert attention_mask[0] == 1
