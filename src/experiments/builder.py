import yaml 

def make_spoof_dataset(dataset_config):
    """These datasets will all be 'spoof' datasets that implement 
    Queryable, Storable, and Preprocessor. 

    Args:
        dataset_config (dict): A dictionary of dataset-related configs. 

    Raises:
        NotImplementedError: The dataset named in the config isn't supported yet. 

    Returns:
        _type_: A dataset wrapper object. 
    """
    if dataset_config['name'].lower() == 'mnist':
        from datasets.mnist_dataset import get_mnist_dataset, MNISTWrapper
        return MNISTWrapper(get_mnist_dataset(version='normal'))
    elif dataset_config['name'].lower() == 'splitmnist':
        from datasets.mnist_dataset import get_mnist_dataset, MNISTWrapper
        return MNISTWrapper(get_mnist_dataset(version='split', configs=dataset_config))
    # elif dataset_config['name'].lower() == 'cifar10':
    #     from datasets.cifar_10_dataset import get_cifar10_dataset
    #     return get_cifar10_dataset(version='normal')
    # elif dataset_config['name'].lower() == 'splitcifar10':
    #     from datasets.cifar_10_dataset import get_cifar10_dataset
    #     return get_cifar10_dataset(version='split', configs=dataset_config)
    else:
        raise NotImplementedError()

def get_dataset(dataset_config):
    if dataset_config['type'] == 'spoof': 
        wrapper = make_spoof_dataset(dataset_config)
        return {
            'queryable': wrapper, 
            'preprocessor': wrapper, 
            'storable': wrapper, 
            'dataset': wrapper, 
        }

def make_model(model_config):
    if model_config['name'].lower() in ['resnet18', 'resnet-18']:
        from torchvision import models
        import torch.nn
        model = models.resnet18(pretrained=False)
        if model_config['in_channels'] != 3:
            model.conv1 = torch.nn.Conv2d(model_config['in_channels'], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
        if model_config['num_classes'] != 1000:
            model.fc = torch.nn.Linear(model.fc.in_features, model_config['num_classes'])
        return model

    if model_config['name'].lower() in ['resnet34', 'resnet-34']:
        from torchvision import models
        import torch.nn
        model = models.resnet34(pretrained=False)
        if model_config['in_channels'] != 3:
            model.conv1 = torch.nn.Conv2d(model_config['in_channels'], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
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

def make_trainer(trainer_configs, model, criterion, optimizer, scheduler_factory, dataset, dataset_configs, num_epochs, device):
    trainer_name = trainer_configs['name']
    trainer_configs.setdefault('get_grad_error', False)
    trainer_configs.setdefault('memory_buffer_size', 10)
    trainer_configs.setdefault('reset_model', True)
    trainer_configs.setdefault('online', False)

    if trainer_name == 'defaultTrainer':
        from trainers.default_trainer import DefaultTrainer
        return DefaultTrainer(model, criterion, optimizer, scheduler_factory, dataset, dataset_configs, num_epochs, device)
    elif trainer_name == 'naiveTaskBasedTrainer':
        from trainers.naive_task_based_trainer import NaiveTaskBasedTrainer
        return NaiveTaskBasedTrainer(model, criterion, optimizer, scheduler_factory, dataset, dataset_configs, num_epochs, device, trainer_configs)
    elif trainer_name == 'uniformSamplingTaskBasedTrainer':
        from trainers.uniform_sampling_task_based_trainer import UniformSamplingTaskBasedTrainer
        return UniformSamplingTaskBasedTrainer(model, criterion, optimizer, scheduler_factory, dataset, dataset_configs, num_epochs, device, trainer_configs)
    elif trainer_name == 'cheaterTaskBasedTrainer':
        from trainers.cheater_task_based_trainer import CheaterTaskBasedTrainer
        return CheaterTaskBasedTrainer(model, criterion, optimizer, scheduler_factory, dataset, dataset_configs, num_epochs, device, trainer_configs)
    elif trainer_name == 'highestLossTaskBasedTrainer':
        from trainers.highest_loss_task_based_trainer import HighestLossTaskBasedTrainer
        return HighestLossTaskBasedTrainer(model, criterion, optimizer, scheduler_factory, dataset, dataset_configs, num_epochs, device, trainer_configs)
    elif trainer_name == 'poolingTaskBasedTrainer':
        from trainers.pooling_task_based_trainer import PoolingTaskBasedTrainer
        return PoolingTaskBasedTrainer(model, criterion, optimizer, scheduler_factory, dataset, dataset_configs, num_epochs, device, trainer_configs)
    elif trainer_name == 'gdumbTrainer':
        from trainers.gdumb_trainer import GDumbTrainer
        return GDumbTrainer(model, criterion, optimizer, scheduler_factory, dataset, dataset_configs, num_epochs, device, trainer_configs)
 
    else:
        raise NotImplementedError()


def get_config(experiment_name, verbose=True):
    with open(f'configs/{experiment_name}.yaml') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        if verbose:
            print('Configs:')
            print(configs)
        return configs