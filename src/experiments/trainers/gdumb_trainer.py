import time
import copy
from tqdm import tqdm 
import torch
from .task_trainer import TaskTrainer
import numpy as np
from datasets.buffer_dataset import BufferDataset

class GDumbTrainer(TaskTrainer):

    def __repr__(self):
        return 'GDumb Trainer'

    def __init__(self, model, criterion, optimizer, scheduler_factory, dataset, dataset_configs, num_epochs, device, trainer_configs):
        super().__init__(model, criterion(), optimizer, scheduler_factory, dataset, dataset_configs, num_epochs, device, trainer_configs)
        self.buffer_dataset = BufferDataset([], [], dataset['train'].augmentation, fake_size=512)

    def train_task(self, task_idx):
        since = time.time()
        train_losses, train_accuracies, gradient_errors = [], [], []

        sampling_loader = torch.utils.data.DataLoader(
            self.dataset['train'], shuffle=True, batch_size=1
        )
        for inputs, label in tqdm(sampling_loader):
            self.gdumb_update(inputs, label.item())

        if task_idx > 0:
            print('Buffer labels: ' + self.buffer.pretty_labels())

        if self.should_reset_model:
            self.reset_model()
            self.reset_scheduler()

        train_loader = torch.utils.data.DataLoader(self.buffer_dataset, shuffle=True, batch_size=self.dataset_configs['batch_size'])   
        train_length = len(self.buffer_dataset)

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
            print('-' * 10)

            self.model.train()

            if self.get_gradient_error:
                true_grad = self.compute_true_grad()

            running_loss = 0.0
            running_corrects = 0

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

                self.scheduler.step()

            epoch_loss = running_loss / train_length
            epoch_acc = running_corrects.double() / train_length

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

    def gdumb_update(self, x, y):
        if not self.buffer.is_full():
            self.buffer.insert_new(x, y)
        else:
            classes, count = np.unique(np.array(self.buffer.bufferY), return_counts=True)
            kc = self.buffer.get_size() / len(classes)
            if y not in classes or count[np.where(classes==y)[0][0]] < kc: # Line 5 of GDumb
                max_label = np.random.choice(np.where(count == count.max())[0]) # Find the class that has the most instances, breaking ties randomly
                idx = np.random.randint(count[max_label])
                to_replace = np.where(self.buffer.bufferY == max_label)[0][idx] # Get a random instance of that class
                self.buffer.insert(to_replace, x, y) # Replace that random instance with (x, y)

        self.buffer_dataset.update(self.buffer)

