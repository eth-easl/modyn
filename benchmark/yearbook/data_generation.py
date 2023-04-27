import argparse
import logging
import os
import pathlib
import pickle

import gdown
import numpy as np
import torch
from torch.utils.data import Dataset


def maybe_download(drive_id, destination_dir, destination_file_name):
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

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description="FMoW Benchmark Storage Script")
    parser_.add_argument(
        "--dir", type=pathlib.Path, action="store", help="Path to data directory"
    )

    return parser_


def main():
    parser = setup_argparser()
    args = parser.parse_args()

    logger.info(f"Downloading data to {args.dir}")

    downloader = YearbookDownloader(args.dir)
    downloader.store_data()

class YearbookDownloader(Dataset):
    time_steps = [i for i in range(1930, 2014)]
    input_dim = (1, 32, 32)
    num_classes = 2
    drive_id = "1mPpxoX2y2oijOvW1ymiHEYd7oMu2vVRb"
    file_name = "yearbook.pkl"

    def __init__(self, data_dir):
        super().__init__()
        maybe_download(
            drive_id=self.drive_id,
            destination_dir=data_dir,
            destination_file_name=self.file_name,
        )
        datasets = pickle.load(open(os.path.join(data_dir, self.file_name), "rb"))
        self._dataset = datasets
        self.data_dir = data_dir

    def _get_year_data(self, year: int):
        images = torch.FloatTensor(
            np.array(
                [
                    img.transpose(2, 0, 1)[0].reshape(*self.input_dim)
                    for img in self._dataset[year][0]["images"]
                ]
            )
        )
        labels = torch.LongTensor(self._dataset[year][0]["labels"])
        return [(images[i], labels[i]) for i in range(len(images))]

    def __len__(self):
        return len(self._dataset["labels"])

    def _get_timestamp(self, year: int) -> int:
        """Yearbook data spans from 1930 to 2013. Since we use os timestamps, each year is mapped to a day starting from
        1/1/1970"""
        DAY_LENGTH_SECONDS = 24 * 60 * 60
        timestamp = ((year - 1930) * DAY_LENGTH_SECONDS) + 1
        return timestamp

    def _create_binary_file(self, data: list[tuple[torch.Tensor, torch.Tensor]], output_file_name: str, year: int):

        with open(output_file_name, "wb") as f:
            for tensor1, tensor2 in data:

                features_bytes = tensor1.numpy().tobytes()

                label_integer = tensor2.item()

                features_size = len(features_bytes)
                assert features_size == 4096

                f.write(int.to_bytes(label_integer, length=4, byteorder="big"))
                f.write(features_bytes)

        timestamp = self._get_timestamp(year)
        os.utime(output_file_name, (timestamp, timestamp))

    def store_data(self):
        # create directories
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        for year in self.time_steps:
            print(f"Saving data for year {year}")
            ds = self._get_year_data(year)
            self._create_binary_file(ds, os.path.join(self.data_dir, f"{year}.bin"), year)

        os.remove(os.path.join(self.data_dir, "yearbook.pkl"))

if __name__ == "__main__":
    main()