import torch
from modyn.models.tokenizers import DistilBertTokenizerTransform


def test_distil_bert_tokenizer_transform():
    max_token_length = 40
    transform = DistilBertTokenizerTransform(max_token_length=max_token_length)

    # Test input string
    input_text = "This is a test sentence."

    # Expected output tensors
    expected_input_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 6251, 1012, 102] + [0] * 32], dtype=torch.int64)
    expected_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1] + [0] * 32], dtype=torch.int64)

    # Call the transform on the input text
    output = transform(input_text)

    # Check if the output has the correct shape and data type
    assert output.shape == (max_token_length, 2)
    assert output.dtype == torch.int64

    # Check if the output contains the expected input_ids and attention_mask tensors
    assert torch.all(torch.eq(output[:, 0], expected_input_ids))
    assert torch.all(torch.eq(output[:, 1], expected_attention_mask))


def test_distil_bert_tokenizer_transform_empty_input():
    max_token_length = 300
    transform = DistilBertTokenizerTransform(max_token_length=max_token_length)

    # Test empty input string
    input_text = ""

    # Call the transform on the empty input text
    output = transform(input_text)

    # Check if the output has the correct shape and data type
    assert output.shape == (max_token_length, 2)
    assert output.dtype == torch.int64

    # Check if the output contains only the [CLS] and [SEP] tokens
    expected_input_ids = torch.tensor([[101, 102] + [0] * 298], dtype=torch.int64)
    expected_attention_mask = torch.tensor([[1, 1] + [0] * 298], dtype=torch.int64)
    assert torch.all(torch.eq(output[:, 0], expected_input_ids))
    assert torch.all(torch.eq(output[:, 1], expected_attention_mask))
