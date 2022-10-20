import time
import copy
from tqdm import tqdm 
import torch
from .task_trainer import TaskTrainer
import numpy as np
from datasets.buffer_dataset import BufferDataset

class UniformSamplingTaskBasedTrainer(TaskTrainer):

    def __repr__(self):
        return 'Uniform Sampling Trainer'

    def __init__(self, model, criterion, optimizer, scheduler, dataset, dataset_configs, num_epochs, device, memory_buffer_size, get_gradient_error):
        super().__init__(model, criterion, optimizer, scheduler, dataset, dataset_configs, num_epochs, device, memory_buffer_size, get_gradient_error)
        self.buffer_dataset = BufferDataset([], [], dataset['train'].augmentation, fake_size=512)

    def validation(self):
        print('Validation')
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        running_count = 0
        self.dataset['test'].set_active_task(0)
        result = {}

        val_loader = torch.utils.data.DataLoader(self.dataset['test'], batch_size = self.dataset_configs['batch_size'], shuffle = False)

        while True:
            print('Task', self.dataset['test'].active_task())
            inner_loss = 0.0
            inner_corrects = 0
            inner_count = 0

            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(False):
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
        all_inputs = []
        all_labels = []

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset([self.dataset['train'], self.buffer_dataset]),
            shuffle=True,
            batch_size=self.dataset_configs['batch_size']
        )

        if task_idx > 0:
            print('Buffer labels: ' + self.buffer.pretty_labels())

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
            print('-' * 10)

            self.model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(train_loader):
                if epoch == 0:
                    all_inputs.append(inputs[(labels==task_idx*2)|(labels==task_idx*2+1)])
                    all_labels.append(labels[(labels==task_idx*2)|(labels==task_idx*2+1)])

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # self.scheduler.step()

            epoch_loss = running_loss / len(self.dataset['train'])
            epoch_acc = running_corrects.double() / len(self.dataset['train'])

            print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        self.update_buffer(task_idx, torch.cat(all_inputs), torch.cat(all_labels))

    def update_buffer(self, task_idx, new_samples, new_labels):
        assert self.memory_buffer_size % 12 == 0, "Prototype only supports memory buffer as multiple of 12. "
        if task_idx >= 4:
            return

        new_sector_size = int(12 / (task_idx+1))

        for i in range(task_idx):
            old_sector_size = int(12 / task_idx)
            # Shrink the i-th task
            for j in range(new_sector_size):
                # self.bufferX[i*new_sector_size+j] = self.bufferX[i*old_sector_size+j]
                # self.bufferY[i*new_sector_size+j] = self.bufferY[i*old_sector_size+j]
                self.buffer.replace(i*new_sector_size+j, i*old_sector_size+j)
        start_idx = task_idx * new_sector_size

        new_indices = np.random.choice(new_labels.shape[0], new_sector_size, replace=False)
        for i in range(new_sector_size):
            # self.bufferX[i+start_idx] = new_samples[new_indices[i]]
            # self.bufferY[i+start_idx] = new_labels[new_indices[i]]
            self.buffer.insert(i+start_idx, new_samples[new_indices[i]], new_labels[new_indices[i]])

        # self.buffer_dataset.update(torch.stack(self.bufferX), self.bufferY)
        self.buffer_dataset.update(self.buffer)
