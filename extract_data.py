import os
import pickle

from wild_time_data import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm

MODELS_PATH = "../new_models"
OUTPUT_FOLDER = "../test_out"


class YearbookNetModel(nn.Module):
    def __init__(self, num_input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            self.conv_block(num_input_channels, 32),
            self.conv_block(32, 32),
            self.conv_block(32, 32),
            self.conv_block(32, 32),
        )
        self.hid_dim = 32
        self.classifier = nn.Linear(32, num_classes)

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2)
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = self.enc(data)
        data = torch.mean(data, dim=(2, 3))

        return self.classifier(data)


class YearbookDataset(Dataset):
    def __init__(self, year):
        data = load_dataset(dataset_name="yearbook", time_step=year, split="train", data_dir="wild-time-data")
        self.samples = [x[0] for x in data]
        self.targets = [x[1] for x in data]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = self.targets[idx]
        return sample, target


def evaluate_model(checkpoint_path, year):
    # Create dataloader
    dataset = YearbookDataset(year)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Load trained model
    model = YearbookNetModel(1, 2)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"])

    # Compute average accuracy
    model.eval()
    accuracies_list = []

    for inputs, targets in dataloader:
        model.eval()
        outputs = model(inputs)
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == targets).float().mean()

        accuracies_list.append(accuracy.item())

    return sum(accuracies_list) / len(accuracies_list)


def get_measurements(experiment_name, delta):
    measurements = []

    for model in tqdm(sorted(os.listdir(os.path.join(MODELS_PATH, experiment_name)))):
        model_path = os.path.join(MODELS_PATH, experiment_name, model)
        model_year = int(model.split(".modyn")[0].split("_")[2]) + delta

        for data_year in range(1930, 2014):
            accuracy = evaluate_model(model_path, data_year)
            measurements.append({
                "model_year": model_year,
                "data_year": data_year,
                "accuracy": accuracy
            })
    return measurements


def get_final_table(measurements):
    lendata = 84
    lenmodels = 83  # the last model is not checkpointed

    # create a matrix in which the cell model_year data_year contains the accuracy for the model trained up that year tested on the given data_year
    data = [0] * lenmodels
    for i in range(lenmodels):
        # create the array for the given model
        model_year = 1930 + i
        data[i] = [0] * lendata
        for sample in measurements:
            if sample["model_year"] == model_year:
                data[i][sample["data_year"] - 1930] = sample["accuracy"]

    return data


def dump_table(experiment_name, data):
    with open(f"{OUTPUT_FOLDER}/{experiment_name}.pkl", "wb") as f:
        pickle.dump(data, f)


def main():

    # make sure that the output folder exists
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    for experiment_name in os.listdir(MODELS_PATH):
        print(f"Processing {experiment_name}")

        # models are sequentially stored using the trigger id. Retrieve the smallest one (first year)
        first_year = min(int(el.split(".modyn")[0].split("_")[2]) for el in os.listdir(os.path.join(MODELS_PATH, experiment_name)))

        # sum this value to the trigger id to get the year
        delta = 1930 - first_year

        measurements = get_measurements(experiment_name, delta)
        data = get_final_table(measurements)

        dump_table(experiment_name, data)


if __name__ == "__main__":
    main()
