from torchvision import models


class ResNet18():
    def __init__(
        self,
        model_configuration,
        device,
    ):

        self.model = models.__dict__['resnet18'](num_classes=model_configuration['num_classes'])
        self._device = device

    def train_one_iteration(self, batch, criterion):

        data, target = batch[0].to(self._device), batch[1].to(self._device)
        output = self.model(data)
        loss = criterion(output, target)
        loss.backward()
