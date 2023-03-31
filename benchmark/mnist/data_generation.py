import argparse
import logging
import os
import pathlib
import random
import shutil
import time

import torch
from PIL import Image
from torchvision.datasets import MNIST

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description="MNIST Benchmark Storage Script")
    parser_.add_argument(
        "--dir", type=pathlib.Path, action="store", help="Path to data directory"
    )
    parser_.add_argument(
        "--timestamps",
        default="RANDOM",
        const="RANDOM",
        choices=["ALLZERO", "INCREASING", "RANDOM"],
        nargs="?",
        help="Parameter to define the timestamps added to the files. \
              ALLZERO (which sets the timestamp of all pngs to zero), \
              INCREASING (which starts at 0 for the first file and then \
              continually increases +1 per file), RANDOM (which sets it to random). Defaults to RANDOM",
    )
    parser_.add_argument(
        "--action",
        default="DOWNLOAD",
        const="DOWNLOAD",
        choices=["DOWNLOAD", "REMOVE"],
        nargs="?",
        help="Define the action taken by the script. \
              DOWNLOAD (download the MNIST dataset into the given dir) or \
              REMOVE (delete all files in the given dir)",
    )

    return parser_


def main():
    parser = setup_argparser()
    args = parser.parse_args()

    if args.action == "DOWNLOAD":
        logger.info(f"Downloading data to {args.dir}")
        _store_data(args.dir, args.timestamps)
    if args.action == "REMOVE":
        logger.info(f"Removing data in {args.dir}")
        _remove_data(args.dir)


def _store_data(data_dir: pathlib.Path, timestamp_option: str):
    # create directories
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # The following line forces a download of the mnist dataset.
    mnist = MNIST(str(data_dir), train=True, download=True)
    # TODO(vGsteiger): Currently leaving out validation set, if #49 is implemented, this should be changed
    x_train = mnist.data.numpy()
    y_train = mnist.targets.numpy()

    # store mnist dataset in png format
    for i, data in enumerate(x_train):
        image = Image.fromarray(data)
        image.save(data_dir / f"{i}.png")
        _set_file_timestamp(data_dir / f"{i}.png", timestamp_option, i)
    for i, label in enumerate(y_train):
        with open(data_dir / f"{i}.label", "w", encoding="utf-8") as file:
            file.write(str(int(label)))
        _set_file_timestamp(data_dir / f"{i}.label", timestamp_option, i)


def _set_file_timestamp(file: str, timestamp_option: str, current: int):
    if timestamp_option == "ALLZERO":
        os.utime(file, (0, 0))
    elif timestamp_option == "INCREASING":
        os.utime(file, (current, current))
    else:
        random_timestamp = random.randint(0, int(round(time.time() * 1000)))
        os.utime(file, (random_timestamp, random_timestamp))


def _remove_data(data_dir: pathlib.Path):
    shutil.rmtree(data_dir)


if __name__ == "__main__":
    main()

