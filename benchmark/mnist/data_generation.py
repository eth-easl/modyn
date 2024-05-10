import os
import random
import shutil
import time
from pathlib import Path
from typing import Annotated, Literal

import typer
from modyn.utils.logging import setup_logging
from PIL import Image
from torchvision.datasets import MNIST

logger = setup_logging(__name__)


def main(
    dir: Annotated[Path, typer.Argument(help="Path to data directory")],
    evaluation: Annotated[
        bool,
        typer.Option(help="Whether to handle training (not present) or evaluation (present) dataset."),
    ] = False,
    timestamps: Annotated[
        Literal["ALLZERO", "INCREASING", "RANDOM"],
        typer.Option(
            help=(
                "Parameter to define the timestamps added to the files. "
                "ALLZERO (which sets the timestamp of all pngs to zero), "
                "INCREASING (which starts at 0 for the first file and then "
                "continually increases +1 per file), RANDOM (which sets it to random)."
            ),
        ),
    ] = "RANDOM",
    action: Annotated[
        Literal["DOWNLOAD", "REMOVE"],
        typer.Option(
            help=(
                "Define the action taken by the script. "
                "DOWNLOAD (download the MNIST dataset into the given dir) or "
                "REMOVE (delete all files in the given dir)"
            ),
        ),
    ] = "DOWNLOAD",
) -> None:
    """MNIST Benchmark Storage Script"""
    if action == "DOWNLOAD":
        logger.info(f"Downloading data to {dir}")
        _store_data(dir, not evaluation, timestamps)
    if action == "REMOVE":
        logger.info(f"Removing data in {dir}")
        shutil.rmtree(dir)


def _store_data(data_dir: Path, train: bool, timestamp_option: str) -> None:
    # create directories
    data_dir.mkdir(exist_ok=True)

    # The following line forces a download of the mnist dataset.
    mnist = MNIST(str(data_dir), train=train, download=True)

    samples = mnist.data.numpy()
    labels = mnist.targets.numpy()

    # store mnist dataset in png format
    for i, data in enumerate(samples):
        image = Image.fromarray(data)
        image.save(data_dir / f"{i}.png")
        if train:
            _set_file_timestamp(data_dir / f"{i}.png", timestamp_option, i)
    for i, label in enumerate(labels):
        with open(data_dir / f"{i}.label", "w", encoding="utf-8") as file:
            file.write(str(int(label)))
        if train:
            _set_file_timestamp(data_dir / f"{i}.label", timestamp_option, i)


def _set_file_timestamp(file: Path, timestamp_option: str, current: int) -> None:
    if timestamp_option == "ALLZERO":
        os.utime(file, (0, 0))
    elif timestamp_option == "INCREASING":
        os.utime(file, (current, current))
    else:
        random_timestamp = random.randint(0, int(round(time.time() * 1000)))
        os.utime(file, (random_timestamp, random_timestamp))


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
