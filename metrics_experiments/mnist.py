import builder
import torch
from torch.optim import lr_scheduler, Adam
import time
import copy 
from tqdm import tqdm
import yaml 
import results_visualizer

if __name__ == '__main__':
    experiment_name = 'exp3'
    write = False

    with open(f'configs/{experiment_name}.yaml') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    dataset = builder.make_dataset(configs['dataset'])
    model = builder.make_model(configs['model'])
    train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size = configs['dataset']['batch_size'], shuffle = True)
    val_loader = torch.utils.data.DataLoader(dataset['test'], batch_size = configs['dataset']['batch_size'], shuffle = False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    epochs = configs['epochs']
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=configs['lr'])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    trainer = builder.make_trainer(configs['trainer'], model, criterion, optimizer, exp_lr_scheduler, 
            {'train': train_loader, 'val': val_loader}, 
            epochs, device)

    result = trainer.train()

    if write: 
        results_visualizer.visualize_results(result, experiment_name)

