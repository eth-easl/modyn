import time
import copy
from tqdm import tqdm 
import torch
from .task_trainer import TaskTrainer
import numpy as np
from datasets.buffer_dataset import BufferDataset

class CheaterTaskBasedTrainer(TaskTrainer):

    def __repr__(self):
        return 'Cheating Trainer (used as a sanity check for debugging)'

    def __init__(self, model, criterion, optimizer, scheduler, dataset, dataset_configs, num_epochs, device, memory_buffer_size, get_gradient_error):
        super().__init__(model, criterion(), optimizer, scheduler, dataset, dataset_configs, num_epochs, device, memory_buffer_size, get_gradient_error)
        self.buffer_dataset = BufferDataset([], [], dataset['train'].augmentation, fake_size=512)

    def validation(self):
        print('Validation')
        self.model.eval()
        self.dataset['test'].set_active_task(0)
        running_loss, running_corrects, running_count, result = 0.0, 0, 0, {}

        val_loader = torch.utils.data.DataLoader(self.dataset['test'], batch_size = self.dataset_configs['batch_size'], shuffle = False)

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

        print('Validation loss: {:.4f} Acc: {:.4f}'.format(all_loss, all_accuracy))
        result['all'] = {
            'Loss': all_loss, 'Accuracy': all_accuracy
        }
        return result 


    def train_task(self, task_idx):
        since = time.time()

        train_loader = torch.utils.data.DataLoader(
            self.dataset['train'], 
            shuffle=True,
            batch_size=self.dataset_configs['batch_size']
        )

        if task_idx > 0:
            print('Buffer labels: ' + self.buffer.pretty_labels())


        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
            print('-' * 10)

            self.model.train()

            if self.get_gradient_error:
                true_grad = self.compute_true_grad()

            running_loss = 0.0
            running_corrects = 0
            running_count = 0
            print('Cheating... setting active task to 0')
            self.dataset['train'].set_active_task(0)
            while True:
                for inputs, labels in tqdm(train_loader):

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(True):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        loss.backward()
                        self.optimizer.step()
                        if self.get_gradient_error: 
                            self.report_grad()  

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    running_count += inputs.size(0)
                
                if not self.next_task():
                    break

                # self.scheduler.step()

            epoch_loss = running_loss / running_count
            epoch_acc = running_corrects.double() / running_count

            if self.get_gradient_error:
                grad_error = self.get_grad_error(true_grad)
                self.clear_grad()
            else:
                grad_error = 'Uncomputed'

            print('Train Loss: {:.4f} Acc: {:.4f}. Gradient error: {:.4f}'.format(epoch_loss, epoch_acc, grad_error))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

