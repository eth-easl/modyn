import builder
import torch
from torch.optim import lr_scheduler, Adam
import time
import copy 
from tqdm import tqdm
import yaml 
import results_visualizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Importance Metrics")
    parser.add_argument("experiment", help="Name of the experiment (for config file and write directory)")
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    experiment_name = args.experiment
    write = args.write

    print(f'Running experiment {experiment_name}. Writing: {"Yes" if write else "No"}')

    with open(f'configs/{experiment_name}.yaml') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        print('Configs:')
        print(configs)

    dataset = builder.make_dataset(configs['dataset'])
    print('Dataset:')
    print(dataset)
    model = builder.make_model(configs['model'])
    print('Model:')
    print(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    epochs = configs['epochs']
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss
    optimizer = Adam(model.parameters(), lr=configs['lr'])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    trainer = builder.make_trainer(configs['trainer'], model, criterion, optimizer, exp_lr_scheduler, 
            dataset, configs['dataset'], 
            epochs, device)

    result = trainer.train()

    print(result)
    if write: 
        results_visualizer.visualize_results(result, experiment_name)

