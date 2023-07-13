import os
import pickle
from typing import Tuple

import numpy as np
import torch
from benchmark_utils import create_fake_timestamp, maybe_download, setup_argparser_wildtime, setup_logger
from torch.utils.data import Dataset

logger = setup_logger()


def main():
    parser = setup_argparser_wildtime("Yearbook")
    args = parser.parse_args()

    logger.info(f"Downloading data to {args.dir}")

    downloader = YearbookDownloader(args.dir)
    downloader.store_data()

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


class YearbookDownloader(Dataset):
    time_steps = [i for i in range(1930, 2014)]
    input_dim = (1, 32, 32)
    num_classes = 2
    drive_id = "1mPpxoX2y2oijOvW1ymiHEYd7oMu2vVRb"
    file_name = "yearbook.pkl"

    def __init__(self, data_dir: str):
        super().__init__()
        maybe_download(
            drive_id=self.drive_id,
            destination_dir=data_dir,
            destination_file_name=self.file_name,
        )
        datasets = pickle.load(open(os.path.join(data_dir, self.file_name), "rb"))
        self._dataset = datasets
        self.data_dir = data_dir

    def _get_year_data(self, year: int) -> list[Tuple]:
        images = torch.FloatTensor(
            np.array(
                [   # transpose to transform from HWC to CHW (H=height, W=width, C=channels).
                    # Pytorch requires CHW format
                    img.transpose(2, 0, 1)[0].reshape(*self.input_dim)
                    # _dataset has 3 dimensions [years][train=0,valid=1]["images"/"labels"]
                    for img in self._dataset[year][0]["images"]
                ]
            )
        )
        labels = torch.LongTensor(self._dataset[year][0]["labels"])
        return [(images[i], labels[i]) for i in range(len(images))]

    def __len__(self) -> int:
        return len(self._dataset["labels"])

    def store_data(self) -> None:
        # create directories
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        for year in self.time_steps:
            print(f"Saving data for year {year}")
            ds = self._get_year_data(year)
            create_binary_file(ds,
                               os.path.join(self.data_dir, f"{year}.bin"),
                               create_fake_timestamp(year, base_year=1930))

        os.remove(os.path.join(self.data_dir, "yearbook.pkl"))


if __name__ == "__main__":
    main()
