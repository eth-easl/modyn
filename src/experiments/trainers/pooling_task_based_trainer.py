import time
import copy
from tqdm import tqdm
import torch
from .task_trainer import TaskTrainer
import numpy as np
from datasets.buffer_dataset import BufferDataset


class PoolingTaskBasedTrainer(TaskTrainer):

    def __repr__(self):
        return '2x2 Pooling Trainer'

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            scheduler,
            dataset,
            dataset_configs,
            num_epochs,
            device,
            trainer_configs):
        super().__init__(
            model,
            criterion(),
            optimizer,
            scheduler,
            dataset,
            dataset_configs,
            num_epochs,
            device,
            trainer_configs)
        self.buffer_dataset = BufferDataset(
            [], [], dataset['train'].augmentation, fake_size=512)

    def train_task(self, task_idx):
        since = time.time()
        all_inputs = []
        all_labels = []
        train_losses, train_accuracies, gradient_errors = [], [], []

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset([self.dataset['train'], self.buffer_dataset]),
            shuffle=True,
            batch_size=self.dataset_configs['batch_size']
        )

        if task_idx > 0:
            print('Buffer labels: ' + self.buffer.pretty_labels())

        if self.should_reset_model:
            self.reset_model()

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, self.num_epochs))
            print('-' * 10)

            self.model.train()

            if self.get_gradient_error:
                true_grad = self.compute_true_grad()

            if self.online and epoch == 1:
                # At this point, you lose the data, so switch the train loader
                # to only the buffer dataset
                train_loader = torch.utils.data.DataLoader(
                    self.buffer_dataset, shuffle=True, batch_size=self.dataset_configs['batch_size'])

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(train_loader):
                if epoch == 0:
                    all_inputs.append(
                        inputs[(labels == task_idx * 2) | (labels == task_idx * 2 + 1)])
                    all_labels.append(
                        labels[(labels == task_idx * 2) | (labels == task_idx * 2 + 1)])

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

                # self.scheduler.step()

            epoch_loss = running_loss / \
                (len(self.dataset['train']) + len(self.buffer_dataset))
            epoch_acc = running_corrects.double(
            ) / (len(self.dataset['train']) + len(self.buffer_dataset))

            if self.get_gradient_error:
                grad_error = self.get_grad_error(true_grad).item()
                self.clear_grad()
            else:
                grad_error = 0

            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc.item())
            gradient_errors.append(grad_error)

            print(
                'Train Loss: {:.4f} Acc: {:.4f}. Gradient error: {:.4f}'.format(
                    epoch_loss, epoch_acc, grad_error))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        self.update_buffer(
            task_idx,
            torch.cat(all_inputs),
            torch.cat(all_labels))

        train_results = {
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
        }
        if self.get_gradient_error:
            train_results['gradient_errors'] = gradient_errors

        return train_results

    def update_buffer(self, task_idx, new_samples, new_labels):
        assert self.memory_buffer_size % 12 == 0, "Prototype only supports memory buffer as multiple of 12. "
        if task_idx >= 4:
            return

        new_samples = torch.nn.MaxPool2d(2, stride=2)(new_samples)
        new_samples = torch.nn.Upsample(
            scale_factor=2, mode='nearest')(new_samples)

        new_sector_size = int(self.memory_buffer_size / (task_idx + 1))

        for i in range(task_idx):
            old_sector_size = int(self.memory_buffer_size / task_idx)
            # Shrink the i-th task
            for j in range(new_sector_size):
                self.buffer.replace(
                    i * new_sector_size + j,
                    i * old_sector_size + j)
        start_idx = task_idx * new_sector_size

        new_indices = np.random.choice(
            new_labels.shape[0], new_sector_size, replace=False)
        for i in range(new_sector_size):
            self.buffer.insert(i + start_idx,
                               new_samples[new_indices[i]],
                               new_labels[new_indices[i]])

        self.buffer_dataset.update(self.buffer)
        self.buffer_dataset.fake_size = int(
            (1 + task_idx) * len(self.dataset['train']))
