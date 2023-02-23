import pathlib

import torch


def instantiate_model():
    pass # do stuff like CUDA, model.eval() here

def load_data(evaluation_data):
    pass

def evaluate_model(model_path: pathlib.Path, evaluation_data: pathlib.Path) -> dict:
    model = instantiate_model()
    
    data, labels = load_data(evaluation_data)

    num_datapoints = len(labels) # might need to adjust
    correct = 0
    with torch.no_grad():
        for batched_data, batched_labels in data: # todo use dataloader here
            output = model(batched_data)
            pred = do_stuff_with(output) # not sure what DLRM output is, need to get output label here
            correct += (pred == batched_labels).float().sum()

    accuracy = correct / num_datapoints

    return { "acc": accuracy }