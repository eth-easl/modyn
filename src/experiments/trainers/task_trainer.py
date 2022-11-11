import time
import copy
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from .buffer import Buffer
import numpy as np


class TaskTrainer:

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            scheduler_factory,
            dataset,
            dataset_configs,
            num_epochs,
            device,
            trainer_configs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler_factory = scheduler_factory
        self.scheduler = scheduler_factory()
        self.dataset = dataset
        self.dataset_configs = dataset_configs
        self.num_epochs = num_epochs
        self.device = device
        self.memory_buffer_size = trainer_configs['memory_buffer_size']
        self.bufferX = [None for _ in range(self.memory_buffer_size)]
        self.bufferY = [None for _ in range(self.memory_buffer_size)]
        self.buffer = Buffer(self.memory_buffer_size)
        self.buffer_dataset = None
        self.get_gradient_error = trainer_configs['get_grad_error']
        self.should_reset_model = trainer_configs['reset_model']
        self.online = trainer_configs['online']
        if self.get_gradient_error:
            self.clear_grad()

        assert dataset['train'].is_task_based()

    def reset_model(self):
        """
        refs:
            - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
            - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
            - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        """
        with torch.no_grad():
            def weight_reset(m):
                # - check if the current module has reset_parameters & if it's callable, call it on m
                reset_parameters = getattr(m, "reset_parameters", None)
                if callable(reset_parameters):
                    m.reset_parameters()

            # Applies fn recursively to every submodule see:
            # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
            self.model.apply(fn=weight_reset)

    def next_task(self):
        if self.dataset['train'].has_more_tasks():
            self.dataset['train'].next_task()
            return True
        return False

    def train(self):
        print('Training with', repr(self))
        num_tasks = len(self.dataset['train'].tasks)
        result = {}
        while True:
            print('Training on task', self.dataset['train'].active_task())
            train_results = self.train_task(self.dataset['train'].active_idx)
            val_results = self.validation()
            result[self.dataset['train'].active_task()] = {
                'train': train_results, 'val': val_results}
            try:
                self.dataset['train'].next_task()
            except IndexError:
                break
        result[f'A{num_tasks}'] = result[self.dataset['train'].tasks[-1]
                                         ]['val']['all']['Accuracy']
        result[f'F{num_tasks}'] = self.get_forgetting(result)
        print('Done!')
        return result

    def get_forgetting(self, result):
        forgetting = []
        for task in self.dataset['train'].tasks:
            # We'll assume that the best is from when we trained on it.
            best = result[task]['val'][task]['Accuracy']
            # And also that the worst is from the last epoch
            worst = result[self.dataset['train'].tasks[-1]
                           ]['val'][task]['Accuracy']
            forgetting.append(best - worst)
        return np.mean(np.array(forgetting))

    def validation(self):
        print('Validation')
        self.model.eval()
        self.dataset['test'].set_active_task(0)
        running_loss, running_corrects, running_count, result = 0.0, 0, 0, {}

        val_loader = torch.utils.data.DataLoader(
            self.dataset['test'],
            batch_size=self.dataset_configs['batch_size'],
            shuffle=False)

        while True:
            print('Task', self.dataset['test'].active_task())
            inner_loss, inner_corrects, inner_count = 0.0, 0, 0

            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                with torch.no_grad():
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_count += inputs.size(0)
                inner_loss += loss.item() * inputs.size(0)
                inner_corrects += torch.sum(preds == labels.data)
                inner_count += inputs.size(0)

            result[self.dataset['test'].active_task()] = {
                'Loss': inner_loss / inner_count, 'Accuracy': inner_corrects.double().item() / inner_count
            }
            try:
                self.dataset['test'].next_task()
            except IndexError:
                break

        all_loss = running_loss / running_count
        all_accuracy = running_corrects.double().item() / running_count

        print(
            'Validation loss: {:.4f} Acc: {:.4f}'.format(
                all_loss, all_accuracy))
        result['all'] = {
            'Loss': all_loss, 'Accuracy': all_accuracy
        }
        return result

    def __repr__(self):
        return 'Generic Task Based Trainer'

    """
        For gradient error visualization, this should be called every SGD step.
    """

    def report_grad(self):
        current_grad = torch.cat([param.grad.flatten()
                                 for param in self.model.parameters()])
        if self.grad_total is None:
            self.grad_total = current_grad
        else:
            self.grad_total += current_grad

    """
        For gradient error visualization, this should be called before each epoch.
        Guarantees that the train dataset will remain on the same active task.
    """

    def compute_true_grad(self):
        self.model.train()
        self.dataset['train'].full_mode = True
        print('Computing true gradient...')
        self.optimizer.zero_grad()
        train_loader = torch.utils.data.DataLoader(
            self.dataset['train'], batch_size=self.dataset_configs['batch_size'])
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.criterion(self.model(inputs), labels).backward()
        true_grad = torch.cat([param.grad.flatten()
                              for param in self.model.parameters()])
        self.optimizer.zero_grad()
        self.dataset['train'].full_mode = False
        print('True gradient computed ')
        return true_grad

    def get_grad_error(self, true_grad):
        print(true_grad - self.grad_total)
        return torch.norm(true_grad - self.grad_total)

    def clear_grad(self):
        self.grad_total = None

    def reset_scheduler(self):
        self.scheduler = self.scheduler_factory()
