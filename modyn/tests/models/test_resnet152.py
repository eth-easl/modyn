import torch
from modyn.models import ResNet152


def test_forward_with_embedding_recording():
    net = ResNet152({"num_classes": 10}, "cpu", False)
    input_data = torch.rand(30, 3, 32, 32)
    output = net.model(input_data)
    assert output.shape == (30, 10)
    assert net.model.embedding is None

    net.model.embedding_recorder.start_recording()
    input_data = torch.rand(30, 3, 32, 32)
    output = net.model(input_data)
    assert output.shape == (30, 10)

    assert net.model.embedding is not None
    assert torch.equal(net.model.fc(net.model.embedding), output)


def test_get_last_layer():
    net = ResNet152({"num_classes": 10}, "cpu", False)
    last_layer = net.model.get_last_layer()

    assert isinstance(last_layer, torch.nn.Linear)
    assert last_layer.in_features == 2048
    assert last_layer.out_features == 10
    assert last_layer.bias.shape == (10,)
    assert last_layer.weight.shape == (10, 2048)
