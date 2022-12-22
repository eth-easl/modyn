from torchvision import models
import torch

class ResNet():
    def __init__(
        self,
        arch,
        torch_optimizer,
        optimizer_args,
        criterion,
        num_classes,
        train_loader,
        val_loader,
        device,
    ):
        self._model = models.__dict__[arch](num_classes=num_classes)
        self._model = self._model.to(device)

        optimizer_func = getattr(torch.optim, torch_optimizer)
        self._optimizer = optimizer_func(self._model.parameters(), **optimizer_args)

        self._criterion = criterion()

        self._train_loader = train_loader
        self._val_loader = val_loader
        self._device = device

    def train(self, num_epochs=1):

        self._model.train()
        for _ in range(num_epochs):

            train_iter = enumerate(self._train_loader)
            for _, batch in train_iter:
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

