import time
import copy
from tqdm import tqdm 
import torch
from torch.utils.data import Dataset
from .buffer import Buffer

class TaskTrainer:

    def __init__(self, model, criterion, optimizer, scheduler, dataset, dataset_configs, num_epochs, device, memory_buffer_size, get_gradient_error):
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
        if self.get_gradient_error:
            self.clear_grad()

        assert dataset['train'].is_task_based() 

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
            self.train_task(self.dataset['train'].active_idx)
            result[self.dataset['train'].active_task()] = self.validation()
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
        if self.grad_avg is None:
            self.grad_avg = current_grad 
            self.num_grads = 1
        else: 
            self.num_grads += 1
            self.grad_avg = (1/self.num_grads)*current_grad+((self.num_grads-1)/self.num_grads)*self.grad_avg

    """
        For gradient error visualization, this should be called before each epoch. Guarantees that the train dataset will remain on the same active task. 
    """
    def compute_true_grad(self):
        original_active_idx = self.dataset['train'].active_idx
        self.model.train()
        self.dataset['train'].set_active_task(0)
        print('Computing true gradient...')
        self.optimizer.zero_grad()
        train_loader = torch.utils.data.DataLoader(self.dataset['train'], batch_size=self.dataset_configs['batch_size'])
        while True:
            for inputs, labels in tqdm(train_loader): 
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.criterion(self.model(inputs), labels).backward()
            if not self.next_task():
                break
        true_grad = torch.cat([param.grad.flatten() for param in self.model.parameters()])
        self.optimizer.zero_grad()
        self.dataset['train'].set_active_task(original_active_idx)
        print('True gradient computed ')
        return true_grad

    def get_grad_error(self, true_grad):
        return torch.norm(true_grad - self.grad_avg)

    def clear_grad(self):
        self.grad_avg = None 
        self.num_grads = 0