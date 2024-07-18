import os
import pathlib
from typing import Annotated

import typer
from PIL import Image
from torchvision.datasets import CIFAR10


def main(
        data_dir: Annotated[pathlib.Path, typer.Argument(help="Path to the data directory")],
):
    print(f"Downloading data to {data_dir}")
    # create directories
    os.makedirs(data_dir, exist_ok=True)

    train_dir = data_dir / "train"

    download_dataset(CIFAR10(str(train_dir), train=True, download=True), train_dir)

    test_dir = data_dir / "test"
    download_dataset(CIFAR10(str(test_dir), train=False, download=True), test_dir)


def download_dataset(dataset: CIFAR10, data_dir: pathlib.Path):
    samples = dataset.data
    labels = dataset.targets
    os.makedirs(data_dir, exist_ok=True)

    # store cifar10 dataset in png format
    for i, data in enumerate(samples):
        image = Image.fromarray(data)
        image_path = data_dir / f"{i}.png"
        image.save(image_path)
        os.utime(image_path, (0, 0))
    for i, label in enumerate(labels):
        with open(data_dir / f"{i}.label", "w", encoding="utf-8") as file:
            label_path = data_dir / f"{i}.label"
            file.write(str(int(label)))
            os.utime(label_path, (0, 0))


if __name__ == "__main__":
    typer.run(main)