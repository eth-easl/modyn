import torch

from modyn.models import YearbookNet


def get_model():
    # Create an instance of the YearbookNetModel with the desired parameters for testing
    num_input_channels = 3
    num_classes = 10
    return YearbookNet({"num_input_channels": num_input_channels, "num_classes": num_classes}, "cpu", False)


def test_model_forward_pass():
    # Create a random input tensor with the appropriate shape
    batch_size = 16
    height = 32
    width = 32
    input_channels = 3
    num_classes = 10
    input_data = torch.randn(batch_size, input_channels, height, width)

    # Perform a forward pass through the model
    model = get_model()
    output = model.model(input_data)

    # Assert that the output has the correct shape
    assert output.shape == (batch_size, num_classes)


def test_model_conv_block():
    # Test the conv_block method of the model

    # Create a random input tensor with the appropriate shape
    batch_size = 16
    height = 32
    width = 32
    input_channels = 3
    input_data = torch.randn(batch_size, input_channels, height, width)

    model = get_model()
    # Get the output of the conv_block method
    output = model.model.conv_block(input_channels, 32)(input_data)

    # Assert that the output has the correct shape
    assert output.shape == (batch_size, 32, height // 2, width // 2)


def test_forward_with_embedding_recording():
    net = YearbookNet({"num_input_channels": 3, "num_classes": 10}, "cpu", False)
    input_data = torch.rand(30, 3, 32, 32)
    output = net.model(input_data)
    assert output.shape == (30, 10)
    assert net.model.embedding is None

    net.model.embedding_recorder.start_recording()
    input_data = torch.rand(30, 3, 32, 32)
    output = net.model(input_data)
    assert output.shape == (30, 10)

    # expected embedding
    expected_embedding = net.model.enc(input_data)
    expected_embedding = torch.mean(expected_embedding, dim=(2, 3))

    assert net.model.embedding is not None
    assert torch.equal(net.model.embedding, expected_embedding)


def test_get_last_layer():
    net = YearbookNet({"num_input_channels": 3, "num_classes": 10}, "cpu", False)
    last_layer = net.model.get_last_layer()

    assert isinstance(last_layer, torch.nn.Linear)
    assert last_layer.in_features == 32
    assert last_layer.out_features == 10
    assert last_layer.bias.shape == (10,)
    assert last_layer.weight.shape == (10, 32)
