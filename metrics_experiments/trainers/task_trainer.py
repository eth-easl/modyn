import time
import copy
from tqdm import tqdm 
import torch
from torch.utils.data import Dataset
from .buffer import Buffer

class TaskTrainer:

    def __init__(self, model, criterion, optimizer, scheduler, dataset, dataset_configs, num_epochs, device, memory_buffer_size, get_gradient_error, reset_model):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset = dataset
        self.dataset_configs = dataset_configs
        self.num_epochs = num_epochs
        self.device = device
        self.memory_buffer_size = memory_buffer_size
        self.bufferX = [None for _ in range(memory_buffer_size)]
        self.bufferY = [None for _ in range(memory_buffer_size)]
        self.buffer = Buffer(memory_buffer_size)
        self.buffer_dataset = None
        self.get_gradient_error = get_gradient_error
        self.should_reset_model = reset_model
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

            # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
            self.model.apply(fn=weight_reset)

    def next_task(self):
        if self.dataset['train'].has_more_tasks():
            self.dataset['train'].next_task()
            return True
        return False

    def train(self):
        print('Training with', repr(self))
        result = {}
        while True:
            print('Training on task', self.dataset['train'].active_task())
            train_results = self.train_task(self.dataset['train'].active_idx)
            val_results = self.validation()
            result[self.dataset['train'].active_task()] = {'train': train_results, 'val': val_results}
            try:
                self.dataset['train'].next_task()
            except IndexError:
                break
        print('Done!')
        return result

    def __repr__(self):
        return 'Generic Task Based Trainer'

    """
        For gradient error visualization, this should be called every SGD step. 
    """
    def report_grad(self):
        current_grad = torch.cat([param.grad.flatten() for param in self.model.parameters()]) 
        if self.grad_total is None:
            self.grad_total = current_grad 
        else: 
            self.grad_total += current_grad

    """
        For gradient error visualization, this should be called before each epoch. Guarantees that the train dataset will remain on the same active task. 
    """
    def compute_true_grad(self):
        self.model.train()
        self.dataset['train'].full_mode = True
        print('Computing true gradient...')
        self.optimizer.zero_grad()
        train_loader = torch.utils.data.DataLoader(self.dataset['train'], batch_size=self.dataset_configs['batch_size'])
        for inputs, labels in tqdm(train_loader): 
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.criterion(self.model(inputs), labels).backward()
        true_grad = torch.cat([param.grad.flatten() for param in self.model.parameters()])
        self.optimizer.zero_grad()
        self.dataset['train'].full_mode = False
        print('True gradient computed ')
        return true_grad

    def get_grad_error(self, true_grad):
        print(true_grad - self.grad_total)
        return torch.norm(true_grad - self.grad_total)

    def clear_grad(self):
        self.grad_total = None 
