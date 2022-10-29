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

    def __init__(self, model, criterion, optimizer, scheduler, dataset, dataset_configs, num_epochs, device, trainer_configs):
        super().__init__(model, criterion(), optimizer, scheduler, dataset, dataset_configs, num_epochs, device, trainer_configs)
        self.buffer_dataset = BufferDataset([], [], dataset['train'].augmentation, fake_size=512)

    def train_task(self, task_idx):
        since = time.time()
        train_losses, train_accuracies, gradient_errors = [], [], []

        train_loader = torch.utils.data.DataLoader(
            self.dataset['train'], 
            shuffle=True,
            batch_size=self.dataset_configs['batch_size']
        )

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
            print('-' * 10)

            self.model.train()

            if self.get_gradient_error:
                true_grad = self.compute_true_grad()

            running_loss = 0.0
            running_corrects = 0
            running_count = 0
            print('Cheating... using full mode')
            self.dataset['train'].full_mode = True
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

                # self.scheduler.step()
            self.dataset['train'].full_mode = False

            epoch_loss = running_loss / running_count
            epoch_acc = running_corrects.double() / running_count

            if self.get_gradient_error:
                grad_error = self.get_grad_error(true_grad).item()
                self.clear_grad()
            else:
                grad_error = 0

            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc.item())
            gradient_errors.append(grad_error)
            print('Train Loss: {:.4f} Acc: {:.4f}. Gradient error: {:.4f}'.format(epoch_loss, epoch_acc, grad_error))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        train_results = {
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
        }
        if self.get_gradient_error:
            train_results['gradient_errors'] = gradient_errors

        return train_results 