from torchvision import models
import torch
import os

class Model():
    def __init__(
        self,
        torch_optimizer,
        optimizer_args,
        model_configuration,
        train_loader,
        val_loader,
        device,
        load_checkpoint_path,
        checkpoint_path,
        checkpoint_interval,
    ):

        print("------- optimizer args: ", optimizer_args)
        print("------- model configuration: ", model_configuration)

        self._model = models.__dict__['resnet18'](num_classes=model_configuration['num_classes'])
        #self._model = self._model.to(device)

        optimizer_func = getattr(torch.optim, torch_optimizer)
        self._optimizer = optimizer_func(self._model.parameters(), **optimizer_args)

        if load_checkpoint_path is not None and os.path.exists(load_checkpoint_path):
            self.load_checkpoint(load_checkpoint_path)

        self._criterion = torch.nn.CrossEntropyLoss()

        self._train_loader = train_loader
        self._val_loader = val_loader
        self._device = device

        self._checkpoint_path = checkpoint_path
        self._checkpoint_interval = checkpoint_interval


    def save_checkpoint(self, iteration):

        # TODO(fotstrt): this might overwrite checkpoints from previous runs
        # we could have a counter for the specific training, and increment it
        # every time a new checkpoint is saved.

        # TODO: we assume a local checkpoint for now,
        # should we add functionality for remote?
        checkpoint_file_name = self._checkpoint_path + f'/model_{iteration}' + '.pt'
        dict_to_save = {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }
        torch.save(dict_to_save, checkpoint_file_name)


    def load_checkpoint(self, path):

        checkpoint_dict = torch.load(path)
        assert 'model' in checkpoint_dict
        assert 'optimizer' in checkpoint_dict
        self._model.load_state_dict(checkpoint_dict['model'])
        self._optimizer.load_state_dict(checkpoint_dict['optimizer'])


    def train(self, num_epochs=1):

        self._model.train()

        self.save_checkpoint(0)

        for _ in range(num_epochs):

             train_iter = enumerate(self._train_loader)
             for i, batch in train_iter:
                 self._optimizer.zero_grad()
                 data, target = batch[0].to(self._device), batch[1].to(self._device)
                 output = self._model(data)
                 loss = self._criterion(output, target)
                 loss.backward()
                 self._optimizer.step()

                 if self._checkpoint_interval > 0 and i % self._checkpoint_interval == 0:
                    self.save_checkpoint(i)


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
