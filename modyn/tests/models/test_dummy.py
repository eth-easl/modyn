import torch
from modyn.models import Dummy


def test_empty_forward():
    net = Dummy({"num_classes": 10}, "cpu", False)
    input_data = torch.rand(30, 3, 32, 32)
    output = net.model(input_data)
    assert output.shape == (30, 3, 32, 32)
    assert torch.equal(output, input_data)
