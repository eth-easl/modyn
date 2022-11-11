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
import dynamicdatasets.offline.offline_server as offline_server
import dynamicdatasets.offline.preprocess.offline_preprocessor as preprocess
from threading import Thread


def parse_args():
    parser = argparse.ArgumentParser(description="Importance Metrics")
    parser.add_argument(
        "experiment",
        help="Name of the experiment (for config file and write directory)")
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    experiment_name = args.experiment
    write = args.write

    print(
        f'Running experiment {experiment_name}. Writing: {"Yes" if write else "No"}')

    with open(f'experiments/configs/{experiment_name}.yaml') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        print('Configs:')
        print(configs)

    with open('config/devel.yaml', 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError
        system_config = parsed_yaml

    dataset_dict = builder.get_dataset(configs['dataset'])

    model = builder.make_model(configs['model'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feeder = feed.Feeder(system_config)
    feeder.connect_task_queriable(dataset_dict['queryable'])

    # Eventually we will want to make this and many othre things into one
    # method call (abstraction hiding)
    Thread(target=lambda: metadata_server.serve(system_config)).start()
    Thread(target=lambda: offline_server.serve(
        system_config, dataset_dict)).start()

    # Note: we can easily create a train_feeder / val_feeder, train/val
    # preprocessor.

    epochs = configs['epochs']
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss
    optimizer = Adam(model.parameters(), lr=configs['lr'])

    def scheduler_factory():
        return lr_scheduler.CosineAnnealingLR(optimizer, 32)

    print('All running!')

    time.sleep(2)

    for i in range(10):
        feeder.task_step()
        time.sleep(1)
        print('Last item: ' + str(preprocessor.get_last_item()))

    # @Viktor: an imagined pseudocode workflow would go like this:
    # strategy = dynamicdatasets.strategies.get('gdumb')
    # while feeder.task_step():
    # (task_step doesn't return anything right now, but it'd do like True if more data, False otherwise. )
        # trainer.continual_learn(model, configs={'max_iters': 1000})
        # trainer.validate()
    # visualize_results(trainer)

    # trainer = builder.make_trainer(configs['trainer'], model, criterion, optimizer, scheduler_factory,
    #         dataset_dict['dataset'], configs['dataset'],
    #         epochs, device)
    # result = trainer.train()
    # print(result)
    # if write:
    #     results_visualizer.visualize_results(result, experiment_name)
