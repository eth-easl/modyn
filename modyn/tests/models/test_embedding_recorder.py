import torch

from modyn.models.coreset_methods_support import EmbeddingRecorder


def test_embedding_recording():
    recorder = EmbeddingRecorder()
    recorder.start_recording()
    input_tensor = torch.tensor([1, 2, 3])
    output_tensor = recorder(input_tensor)
    assert torch.equal(recorder.embedding, input_tensor)
    assert torch.equal(output_tensor, input_tensor)


def test_no_embedding_recording():
    recorder = EmbeddingRecorder()
    input_tensor = torch.tensor([4, 5, 6])
    output_tensor = recorder(input_tensor)
    assert recorder.embedding is None
    assert torch.equal(output_tensor, input_tensor)


def test_toggle_embedding_recording():
    recorder = EmbeddingRecorder()
    recorder.start_recording()
    input_tensor = torch.tensor([7, 8, 9])
    output_tensor = recorder(input_tensor)
    assert torch.equal(recorder.embedding, input_tensor)
    assert torch.equal(output_tensor, input_tensor)
    recorder.end_recording()
    input_tensor = torch.tensor([10, 11, 12])
    output_tensor = recorder(input_tensor)
    assert recorder.embedding is None
    assert torch.equal(output_tensor, input_tensor)
