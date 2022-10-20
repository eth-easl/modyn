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