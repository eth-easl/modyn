import time
import copy
from tqdm import tqdm 
import torch
from .task_trainer import TaskTrainer
import numpy as np
from datasets.buffer_dataset import BufferDataset

class HighestLossTaskBasedTrainer(TaskTrainer):

    def __repr__(self):
        return 'Highest Loss Trainer'

    def __init__(self, model, criterion, optimizer, scheduler, dataset, dataset_configs, num_epochs, device, memory_buffer_size, get_gradient_error, reset_model):
        super().__init__(model, criterion(), optimizer, scheduler, dataset, dataset_configs, num_epochs, device, memory_buffer_size, get_gradient_error, reset_model)
        self.criterion_noreduce = criterion(reduction='none')
        self.buffer_dataset = BufferDataset([], [], dataset['train'].augmentation, fake_size=512)

    # def validation(self):
    #     print('Validation')
    #     self.model.eval()
    #     running_loss = 0.0
    #     running_corrects = 0
    #     running_count = 0
    #     self.dataset['test'].set_active_task(0)
    #     result = {}

    #     val_loader = torch.utils.data.DataLoader(self.dataset['test'], batch_size = self.dataset_configs['batch_size'], shuffle = False)

    #     while True:
    #         print('Task', self.dataset['test'].active_task())
    #         inner_loss = 0.0
    #         inner_corrects = 0
    #         inner_count = 0

    #         for inputs, labels in tqdm(val_loader):
    #             inputs = inputs.to(self.device)
    #             labels = labels.to(self.device)

    #             self.optimizer.zero_grad()

    #             with torch.set_grad_enabled(False):
    #                 outputs = self.model(inputs)
    #                 _, preds = torch.max(outputs, 1)
    #                 loss_each = self.criterion(outputs, labels)
    #                 loss = torch.mean(loss_each)

    #             # statistics
    #             running_loss += loss.item() * inputs.size(0)
    #             running_corrects += torch.sum(preds == labels.data)
    #             running_count += inputs.size(0)

    #             inner_loss += loss.item() * inputs.size(0)
    #             inner_corrects += torch.sum(preds == labels.data)
    #             inner_count += inputs.size(0)

    #         result[self.dataset['test'].active_task()] = {
    #             'Loss': inner_loss / inner_count, 'Accuracy': inner_corrects.double().item() / inner_count
    #         }
    #         try:
    #             self.dataset['test'].next_task()
    #         except IndexError:
    #             break

    #     all_loss = running_loss / running_count
    #     all_accuracy = running_corrects.double().item() / running_count

    #     print('Validation loss: {:.4f} Acc: {:.4f}'.format(all_loss, all_accuracy))
    #     result['all'] = {
    #         'Loss': all_loss, 'Accuracy': all_accuracy
    #     }
    #     return result 


    def train_task(self, task_idx):
        since = time.time()
        train_losses, train_accuracies, gradient_errors = [], [], []

        all_inputs = []
        all_labels = []

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
            print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
            print('-' * 10)

            if self.get_gradient_error:
                true_grad = self.compute_true_grad()

            self.model.train()
            running_loss = 0.0
            running_corrects = 0

            inputs_all = []
            labels_all = []
            losses_all = []

            for inputs, labels in tqdm(train_loader):
                if epoch == self.num_epochs - 1:
                    all_inputs.append(inputs[(labels==task_idx*2)|(labels==task_idx*2+1)])
                    all_labels.append(labels[(labels==task_idx*2)|(labels==task_idx*2+1)])

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss_each = self.criterion_noreduce(outputs, labels)
                    loss = torch.mean(loss_each)
                    loss.backward()
                    self.optimizer.step()

                    if self.get_gradient_error: 
                        self.report_grad()  

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                inputs_all.append(inputs)
                labels_all.append(labels)
                if epoch == self.num_epochs - 1:
                    losses_all.append(loss_each[(labels==task_idx*2)|(labels==task_idx*2+1)])

                # self.scheduler.step()

            if self.get_gradient_error:
                grad_error = self.get_grad_error(true_grad).item()
                self.clear_grad()
            else:
                grad_error = 0

            epoch_loss = running_loss / (len(self.dataset['train']) + len(self.buffer_dataset))
            epoch_acc = running_corrects.double() / (len(self.dataset['train']) + len(self.buffer_dataset))

            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc.item())
            gradient_errors.append(grad_error)

            print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        self.update_buffer(task_idx, torch.cat(all_inputs), torch.cat(all_labels), torch.cat(losses_all))

        train_results = {
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
        }
        if self.get_gradient_error:
            train_results['gradient_errors'] = gradient_errors

        return train_results 

    def update_buffer(self, task_idx, new_samples, new_labels, all_losses):
        assert self.memory_buffer_size % 12 == 0, "Prototype only supports memory buffer as multiple of 12. "
        if task_idx >= 4:
            return

        new_sector_size = int(self.memory_buffer_size / (task_idx+1))

        for i in range(task_idx):
            old_sector_size = int(self.memory_buffer_size / task_idx)
            # Shrink the i-th task
            for j in range(new_sector_size):
                self.buffer.replace(i*new_sector_size+j, i*old_sector_size+j)
        start_idx = task_idx * new_sector_size

        new_indices = torch.topk(all_losses, new_sector_size, largest=False, sorted=True).indices
        # new_indices = np.random.choice(new_labels.shape[0], new_sector_size, replace=False)
        for i in range(new_sector_size):
            self.buffer.insert(i+start_idx, new_samples[new_indices[i]], new_labels[new_indices[i]])

        self.buffer_dataset.update(self.buffer)
        self.buffer_dataset.fake_size = (1+task_idx) * len(self.dataset['train'])
        print(self.buffer_dataset.fake_size)

