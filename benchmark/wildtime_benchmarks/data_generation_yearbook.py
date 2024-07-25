"""Create the Yearbook dataset from the WildTime benchmark.

Note:
As modyn operates on unix timestamps (seconds since 1970), we need to create a fake timestamp for the yearbook dataset.
We do this by converting the year to days since epoch (1970-01-01).
"""

import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from benchmark.wildtime_benchmarks.benchmark_utils import (
    create_fake_timestamp,
    download_if_not_exists,
    setup_argparser_wildtime,
    setup_logger,
)
from torch.utils.data import Dataset

logger = setup_logger()


def main() -> None:
    parser = setup_argparser_wildtime("Yearbook", all_arg=False)
    args = parser.parse_args()

    logger.info(f"Downloading data to {args.dir}")

    downloader = YearbookDownloader(args.dir)
    downloader.store_data(args.dummyyear, args.customsplit)


class YearbookDownloader(Dataset):
    time_steps = [i for i in range(1930, 2014)]
    num_classes = 2
    drive_id = "1mPpxoX2y2oijOvW1ymiHEYd7oMu2vVRb"
    file_name = "yearbook.pkl"

    def __init__(self, data_dir: Path):
        super().__init__()
        download_if_not_exists(
            drive_id=self.drive_id,
            destination_dir=data_dir,
            destination_file_name=self.file_name,
        )
        datasets = pickle.load(open(data_dir / self.file_name, "rb"))
        self._dataset = datasets
        self.data_dir = data_dir

    def _get_year_data(self, year: int) -> tuple[dict[str, list[tuple]], dict[str, int]]:
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

        train_size = len(get_split_by_id(0))
        test_size = len(get_split_by_id(1))
        all_size = len(get_split_by_id(2))
        ds = {"train": get_split_by_id(0), "test": get_split_by_id(1), "all": get_split_by_id(2)}
        stats = {"train": train_size, "test": test_size, "all": all_size}
        return ds, stats

    def __len__(self) -> int:
        return len(self._dataset["labels"])
    
    def generate_custom_split(self):
        print("Generating custom yearbook split!")
        # Merge train and test datasets
        merged_data = {}
        for year in self.time_steps:
            year_data, _ = self._get_year_data(year)
            train_data = year_data['train']
            test_data = year_data['test']
            merged_data[year] = train_data + test_data
            np.random.shuffle(merged_data[year])  # Shuffle the merged dataset

        # Generate new train/test split
        for year in self.time_steps:
            year_data, _ = self._get_year_data(year)
            original_test_size = len(year_data['test'])
            self._dataset[year]['test'] = merged_data[year][:original_test_size]
            self._dataset[year]['train'] = merged_data[year][original_test_size:]

    def store_data(self, add_final_dummy_year: bool, custom_split: bool) -> None:
        # create directories
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        split_dirs = {
            name: self.data_dir / name
            for name in ["train", "test", "all"]
        }
        for dir_ in split_dirs.values():
            os.makedirs(dir_, exist_ok=True)

        if custom_split:
            self.generate_custom_split()

        overall_stats = {}
        for year in self.time_steps:
            ds, stats = self._get_year_data(year)
            overall_stats[year] = stats
            for split, split_dir in split_dirs.items():
                self.create_binary_file(
                    ds[split], split_dir / f"{year}.bin", create_fake_timestamp(year, base_year=1930)
                )

        with open(self.data_dir / "overall_stats.json", "w") as f:
            import json
            json.dump(overall_stats, f, indent=4)

        if add_final_dummy_year:
            dummy_year = year + 1
            dummy_data = [ ds["train"][0] ] # get one sample from the previous year
            for split_dir in split_dirs.values():
                self.create_binary_file(
                    dummy_data, split_dir / f"{dummy_year}.bin", create_fake_timestamp(dummy_year, base_year=1930)
                )

    @staticmethod
    def create_binary_file(data: list[tuple], output_file_name: Path, timestamp: int) -> None:
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
