def make_dataset(dataset_config):
    if dataset_config['name'].lower() == 'mnist':
        from datasets.mnist_dataset import get_mnist_dataset
        return get_mnist_dataset(version='normal')
    elif dataset_config['name'].lower() == 'splitmnist':
        from datasets.mnist_dataset import get_mnist_dataset
        return get_mnist_dataset(version='split', collapse_targets=dataset_config['collapse_targets'])
    elif dataset_config['name'].lower() == 'cifar10':
        from datasets.cifar_10_dataset import get_cifar10_dataset
        return get_cifar10_dataset(version='normal')
    elif dataset_config['name'].lower() == 'splitcifar10':
        from datasets.cifar_10_dataset import get_cifar10_dataset
        return get_cifar10_dataset(version='split', collapse_targets=dataset_config['collapse_targets'])
    else:
        raise NotImplementedError()

def make_model(model_config):
    if model_config['name'].lower() in ['resnet18', 'resnet-18']:
        from torchvision import models
        import torch.nn
        model = models.resnet18(pretrained=False)
        if model_config['input_channels'] != 3:
            model.conv1 = torch.nn.Conv2d(model_config['input_channels'], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
        if model_config['num_classes'] != 1000:
            model.fc = torch.nn.Linear(model.fc.in_features, model_config['num_classes'])
        return model
    
    elif model_config['name'].lower() == 'vanilla':
        from models.fc import FCNet
        model = FCNet(model_config)
        return model 

    elif model_config['name'].lower() == 'smallconv':
        from models.small_conv import SmallConv
        return SmallConv(model_config)

    else:
        raise NotImplementedError()

def make_trainer(trainer_configs, model, criterion, optimizer, scheduler, dataset, dataset_configs, num_epochs, device):
    trainer_name = trainer_configs['name']
    memory_buffer_size = trainer_configs.get('memory_buffer_size', 10)

    if trainer_name == 'defaultTrainer':
        from trainers.default_trainer import DefaultTrainer
        return DefaultTrainer(model, criterion, optimizer, scheduler, dataset, dataset_configs, num_epochs, device)
    elif trainer_name == 'naiveTaskBasedTrainer':
        from trainers.naive_task_based_trainer import NaiveTaskBasedTrainer
        return NaiveTaskBasedTrainer(model, criterion, optimizer, scheduler, dataset, dataset_configs, num_epochs, device, memory_buffer_size)
    elif trainer_name == 'uniformSamplingTaskBasedTrainer':
        from trainers.uniform_sampling_task_based_trainer import UniformSamplingTaskBasedTrainer
        return UniformSamplingTaskBasedTrainer(model, criterion, optimizer, scheduler, dataset, dataset_configs, num_epochs, device, memory_buffer_size)
    elif trainer_name == 'highestLossTaskBasedTrainer':
        from trainers.highest_loss_task_based_trainer import HighestLossTaskBasedTrainer
        return HighestLossTaskBasedTrainer(model, criterion, optimizer, scheduler, dataset, dataset_configs, num_epochs, device, memory_buffer_size)
    else:
        raise NotImplementedError()
