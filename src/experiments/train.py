import builder
import torch
from torch.optim import lr_scheduler, Adam
import time
import copy 
from tqdm import tqdm
import yaml 
import results_visualizer
import argparse

import dynamicdatasets.feeder.feeder as feed
import dynamicdatasets.metadata.metadata_server as metadata_server
import dynamicdatasets.offline.offline_preprocessor as preprocess
from threading import Thread 

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

    dataset_dict = builder.make_dataset(configs['dataset'])

    model = builder.make_model(configs['model'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feeder = feed.Feeder()
    feeder.connect_task_queriable(dataset_dict['queryable'])
    Thread(target=lambda: feeder.run()).start()

    preprocessor = preprocess.OfflinePreprocessor()
    preprocessor.set_preprocess(dataset_dict['preprocessor'].preprocess)
    preprocessor.set_storable(dataset_dict['storable'])

    Thread(target=lambda: metadata_server.serve()).start()
    Thread(target=lambda: preprocessor.main()).start()

    epochs = configs['epochs']
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss
    optimizer = Adam(model.parameters(), lr=configs['lr'])
    def scheduler_factory():
        return lr_scheduler.CosineAnnealingLR(optimizer, 32)

    trainer = builder.make_trainer(configs['trainer'], model, criterion, optimizer, scheduler_factory, 
            dataset_dict['dataset'], configs['dataset'], 
            epochs, device)

            

    result = trainer.train()

    print(result)
    if write: 
        results_visualizer.visualize_results(result, experiment_name)

