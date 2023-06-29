import argparse
import logging
import os
import pathlib
from datetime import datetime

import gdown


def maybe_download(drive_id: str, destination_dir: str, destination_file_name: str) -> None:
    """
    Function to download data from Google Drive. Used for Wild-time based benchmarks.
    This function is adapted from wild-time-data
    """
    destination_dir = pathlib.Path(destination_dir)
    destination = destination_dir / destination_file_name
    if destination.exists():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    gdown.download(
        url=f"https://drive.google.com/u/0/uc?id={drive_id}&export=download&confirm=pbef",
        output=str(destination),
        quiet=False,
    )

def setup_argparser_wildtime(dataset: str) -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description=f"{dataset} Benchmark Storage Script")
    parser_.add_argument(
        "--dir", type=pathlib.Path, action="store", help="Path to data directory"
    )

    return parser_

def setup_logger():
    logging.basicConfig(
        level=logging.NOTSET,
        format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )
    return logging.getLogger(__name__)


def create_binary_file(data, output_file_name: str, timestamp: int) -> None:

    with open(output_file_name, "wb") as f:
        for tensor1, tensor2 in data:
            features_bytes = tensor1.numpy().tobytes()
            label_integer = tensor2.item()

            features_size = len(features_bytes)
            assert features_size == 4096

            f.write(int.to_bytes(label_integer, length=4, byteorder="big"))
            f.write(features_bytes)

    os.utime(output_file_name, (timestamp, timestamp))


DAY_LENGTH_SECONDS = 24 * 60 * 60
def create_fake_timestamp(year: int, base_year: int) -> int:
    timestamp = ((year - base_year) * DAY_LENGTH_SECONDS) + 1
    return timestamp

def create_timestamp(year: int, month: int = 1, day: int = 1) -> int:
    return int(datetime(year=year, month=month, day=day).timestamp())