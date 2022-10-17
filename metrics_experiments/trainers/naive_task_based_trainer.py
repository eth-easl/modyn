import time
import copy
from tqdm import tqdm 
import torch
from .task_trainer import TaskTrainer

class NaiveTaskBasedTrainer(TaskTrainer):

    def __init__(self, model, criterion, optimizer, scheduler, dataset, dataset_configs, num_epochs, device, memory_buffer_size):
        super().__init__(model, criterion, optimizer, scheduler, dataset, dataset_configs, num_epochs, device, memory_buffer_size)


    def train(self):
        print("naive task based training!")
        result = {}
        while True:
            print('Training on task', self.dataset['train'].active_task())
            self.train_task()
            result[self.dataset['train'].active_task()] = self.validation()
            try:
                self.dataset['train'].next_task()
            except IndexError:
                break
        print('Done!')
        return result

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


    def train_task(self, ):
        since = time.time()
        train_loader = torch.utils.data.DataLoader(self.dataset['train'], batch_size = self.dataset_configs['batch_size'], shuffle = True)
        
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
            print('-' * 10)

            self.model.train()
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

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # self.scheduler.step()

            epoch_loss = running_loss / len(self.dataset['train'])
            epoch_acc = running_corrects.double() / len(self.dataset['train'])

            print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
