import argparse
import logging
import os
import pathlib
import pickle
from datetime import datetime

import numpy as np
import torch
from benchmark.download_utils import maybe_download
from torch.utils.data import Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

def get_timestamp(year) -> int:
    return int(datetime(year=year, month=1, day=1).timestamp())

def get_fake_timestamp(year: int) -> int:
    DAY_LENGTH_SECONDS = 24 * 60 * 60
    timestamp = ((year - 2007) * DAY_LENGTH_SECONDS) + 1
    return timestamp

def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description="Arxiv Benchmark Storage Script")
    parser_.add_argument(
        "--dir", type=pathlib.Path, action="store", help="Path to data directory"
    )

    return parser_


def main():
    parser = setup_argparser()
    args = parser.parse_args()

    logger.info(f"Downloading data to {args.dir}")
    ArXivDownloader(args.dir).store_data()



class ArXivDownloader(Dataset):
    time_steps = [i for i in range(2007, 2023)]
    input_dim = 55
    num_classes = 172
    drive_id = "1H5xzHHgXl8GOMonkb6ojye-Y2yIp436V"
    file_name = "arxiv.pkl"

    def __getitem__(self, idx):
        return self._dataset["title"][idx], torch.LongTensor([self._dataset["category"][idx]])[0]

    def __len__(self):
        return len(self._dataset["category"])

    def __init__(self,  data_dir):
        super().__init__()

        maybe_download(
            drive_id=self.drive_id,
            destination_dir=data_dir,
            destination_file_name=self.file_name,
        )
        datasets = pickle.load(open(os.path.join(data_dir, self.file_name), "rb"))
        assert self.time_steps == list(sorted(datasets.keys()))
        self._dataset = datasets
        self.path = data_dir

    def store_data(self):
        counter = 0
        for year in tqdm(self._dataset):
            year_timestamp = get_fake_timestamp(year)
            for i in range(len(self._dataset[year][0]["title"])):
                text = self._dataset[year][0]["title"][i]
                label = self._dataset[year][0]["category"][i]

                #store the sentence
                text_file = os.path.join(self.path, f"{counter}.txt")
                with open(text_file, "wb") as f:
                    np.save(f, text)

                #set timestamp
                os.utime(text_file, (year_timestamp, year_timestamp))

                #store the labels
                label_file = os.path.join(self.path, f"{counter}.label")
                with open(label_file, "w", encoding="utf-8") as f:
                    f.write(str(int(label)))

                #set timestamp
                os.utime(label_file, (year_timestamp, year_timestamp))
                counter+=1
        os.remove(os.path.join(self.path, "arxiv.pkl"))


if __name__ == "__main__":
    main()