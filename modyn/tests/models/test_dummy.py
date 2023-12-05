import torch
from modyn.models import Dummy


def test_empty_forward():
    net = Dummy({"num_classes": 10}, "cpu", False)
    input_data = torch.rand(10, 1, 2)
    output = net.model(input_data)
    assert output.shape == (10, 1, 2)
    assert torch.equal(output, net.model.output(input_data))
