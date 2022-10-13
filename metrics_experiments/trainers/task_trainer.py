import time
import copy
from tqdm import tqdm 
import torch

class TaskTrainer:

    def __init__(self, model, criterion, optimizer, scheduler, dataloaders, num_epochs, device, memory_buffer_size):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.num_epochs = num_epochs
        self.device = device
        self.memory_buffer_size = memory_buffer_size
        self.buffer = [None for _ in range(memory_buffer_size)]
        assert dataloaders['train'].dataset.is_task_based() 

    def next_task(self):
        if self.dataloaders['train'].dataset.has_more_tasks():
            self.dataloaders['train'].dataset.next_task()
            return True
        return False

    def train(self):
        raise NotImplementedError()