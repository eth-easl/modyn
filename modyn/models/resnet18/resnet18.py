from torchvision import models
import torch

from modyn.models.base_trainer import BaseTrainer


class ResNet18(BaseTrainer):
    def __init__(
        self,
        torch_optimizer,
        optimizer_args,
        model_configuration,
        train_loader,
        val_loader,
        device,
        checkpoint_path,
        checkpoint_interval,
    ):

        super().__init__(train_loader, val_loader, device, checkpoint_path, checkpoint_interval)

        print("------- optimizer args: ", optimizer_args)
        print("------- model configuration: ", model_configuration)

        self._model = models.__dict__['resnet18'](num_classes=model_configuration['num_classes'])
        self._model = self._model.to(device)

        optimizer_func = getattr(torch.optim, torch_optimizer)
        self._optimizer = optimizer_func(self._model.parameters(), **optimizer_args)

        self._criterion = torch.nn.CrossEntropyLoss()

    def train_one_iteration(self, iteration, batch):

        self._optimizer.zero_grad()
        data, target = batch[0].to(self._device), batch[1].to(self._device)
        output = self._model(data)
        loss = self._criterion(output, target)
        loss.backward()
        self._optimizer.step()

    def evaluate(self):

        self._model.eval()
        running_loss = 0.0

        val_iter = enumerate(self._val_loader)

        correct = 0
        total = 0

        for _, batch in val_iter:
            self._optimizer.zero_grad()
            data, target = batch[0].to(self._device), batch[1].to(self._device)
            with torch.no_grad():
                output = self._model(data)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            loss = self._criterion(output, target)
            running_loss += loss.item() * data.size(0)

        print(f'Accuracy is: {correct/total}')
