import os
import pickle
from typing import Tuple

import numpy as np
import torch
from benchmark_utils import create_fake_timestamp, download_if_not_exists, setup_argparser_wildtime, setup_logger
from torch.utils.data import Dataset

logger = setup_logger()


def main():
    parser = setup_argparser_wildtime("Yearbook")
    args = parser.parse_args()

    logger.info(f"Downloading data to {args.dir}")

    downloader = YearbookDownloader(args.dir)
    downloader.store_data(args.all, args.dummyyear)


class YearbookDownloader(Dataset):
    time_steps = [i for i in range(1930, 2014)]
    num_classes = 2
    drive_id = "1mPpxoX2y2oijOvW1ymiHEYd7oMu2vVRb"
    file_name = "yearbook.pkl"

    def __init__(self, data_dir: str):
        super().__init__()
        download_if_not_exists(
            drive_id=self.drive_id,
            destination_dir=data_dir,
            destination_file_name=self.file_name,
        )
        datasets = pickle.load(open(os.path.join(data_dir, self.file_name), "rb"))
        self._dataset = datasets
        self.data_dir = data_dir

    def _get_year_data(self, year: int, create_test_data: bool) -> tuple[dict[str, list[tuple]], dict[str, int]]:
        def get_split_by_id(split: int) -> list[Tuple]:
            images = torch.FloatTensor(
                np.array(
                    [   # transpose to transform from HWC to CHW (H=height, W=width, C=channels).
                        # Pytorch requires CHW format
                        img.transpose(2, 0, 1)
                        # _dataset has 2 dimensions [years][train=0,test=1]["images"/"labels"]
                        for img in self._dataset[year][split]["images"]
                    ]
                )
            )
            labels = torch.LongTensor(self._dataset[year][split]["labels"])
            return [(images[i], labels[i]) for i in range(len(images))]

        if not create_test_data:
            train_size = len(get_split_by_id(0))
            ds = {"train": get_split_by_id(0)}
            stats = { "train": train_size }
        else:
            train_size = len(get_split_by_id(0))
            test_size = len(get_split_by_id(1))
            ds = {"train": get_split_by_id(0), "test": get_split_by_id(1)}
            stats = {"train": train_size, "test": test_size}
        return ds, stats

    def __len__(self) -> int:
        return len(self._dataset["labels"])

    def store_data(self, create_test_data: bool, add_final_dummy_year: bool) -> None:
        # create directories
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        train_dir = os.path.join(self.data_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        if create_test_data:
            test_dir = os.path.join(self.data_dir, "test")
            os.makedirs(test_dir, exist_ok=True)

        overall_stats = {}
        for year in self.time_steps:
            ds, stats = self._get_year_data(year, create_test_data)
            overall_stats[year] = stats
            self.create_binary_file(ds["train"],
                                    os.path.join(train_dir, f"{year}.bin"),
                                    create_fake_timestamp(year, base_year=1930))
            if create_test_data:
                self.create_binary_file(ds["test"],
                                        os.path.join(test_dir, f"{year}.bin"),
                                        create_fake_timestamp(year, base_year=1930))

        with open(os.path.join(self.data_dir, "overall_stats.json"), "w") as f:
            import json
            json.dump(overall_stats, f, indent=4)

        if add_final_dummy_year:
            dummy_year = year + 1
            dummy_data = [ ds["train"][0] ] # get one sample from the previous year
            self.create_binary_file(dummy_data,
                                    os.path.join(train_dir, f"{dummy_year}.bin"),
                                    create_fake_timestamp(dummy_year, base_year=1930))
            if create_test_data:
                self.create_binary_file(dummy_data,
                                        os.path.join(test_dir, f"{dummy_year}.bin"),
                                        create_fake_timestamp(dummy_year, base_year=1930))

        os.remove(os.path.join(self.data_dir, "yearbook.pkl"))

    @staticmethod
    def create_binary_file(data, output_file_name: str, timestamp: int) -> None:
        with open(output_file_name, "wb") as f:
            for tensor1, tensor2 in data:
                features_bytes = tensor1.numpy().tobytes()
                label_integer = tensor2.item()

                features_size = len(features_bytes)
                assert features_size == 12288

                f.write(int.to_bytes(label_integer, length=4, byteorder="big"))
                f.write(features_bytes)

        os.utime(output_file_name, (timestamp, timestamp))


if __name__ == "__main__":
    main()
