from torchvision import models


class ResNet18():
    def __init__(
        self,
        model_configuration,
    ):

        self.model = models.__dict__['resnet18'](num_classes=model_configuration['num_classes'])
