import os
import pickle
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from benchmark.wildtime_benchmarks.benchmark_utils import (
    create_timestamp,
    download_if_not_exists,
    setup_argparser_wildtime,
    setup_logger,
)

logger = setup_logger()


def main():
    parser = setup_argparser_wildtime("Huffpost", all_arg=False)
    parser.add_argument(
        "--mode",
        choices=["train", "all", "train_and_test"],
        help=(
            "Weather to store only training data (`train`), training and testing (`train_and_test`), or merge "
            "and only have one split (`all`)"
        ),
    )
    args = parser.parse_args()

    logger.info(f"Downloading data to {args.dir}")
    hp = HuffpostDownloader(args.dir)
    hp.store_data(args.mode, args.dummyyear)


class HuffpostDownloader(Dataset):
    time_steps = [i for i in range(2012, 2019)]
    input_dim = 44
    num_classes = 11
    drive_id = "1jKqbfPx69EPK_fjgU9RLuExToUg7rwIY"  # spellchecker:disable-line
    file_name = "huffpost.pkl"

    def __getitem__(self, idx):
        return self._dataset["title"][idx], torch.LongTensor([self._dataset["category"][idx]])[0]

    def __len__(self):
        return len(self._dataset["category"])

    def __init__(self, data_dir: str):
        super().__init__()

        download_if_not_exists(
            drive_id=self.drive_id,
            destination_dir=data_dir,
            destination_file_name=self.file_name,
        )
        datasets = pickle.load(open(os.path.join(data_dir, self.file_name), "rb"))
        assert self.time_steps == list(sorted(datasets.keys()))
        self._dataset = datasets
        self.path = Path(data_dir)

    def store_data(self, mode: Literal["train", "all", "train_and_test"], add_final_dummy_year: bool) -> None:
        for year in tqdm(self._dataset):
            year_timestamp = create_timestamp(year=1970, month=1, day=year - 2011)
            year_rows: dict[str, list] = {"train": [], "test": [], "all": []}
            splits = [0, 1]
            for split in splits:
                if mode == "train" and split != 0:
                    continue
                for i in range(len(self._dataset[year][split]["headline"])):
                    text = self._dataset[year][split]["headline"][i]
                    label = self._dataset[year][split]["category"][i]
                    csv_row = f"{text}\t{label}"
                    if mode == "train":
                        year_rows["train"].append(csv_row)
                    elif mode == "all":
                        year_rows["all"].append(csv_row)
                    elif mode == "train_and_test":
                        year_rows["train" if split == 0 else "test"].append(csv_row)

            if mode == "train" or mode == "train_and_test":
                # store the sentences
                (self.path / "train").mkdir(parents=True, exist_ok=True)
                text_file = self.path / "train" / f"{year}.csv"
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(year_rows["train"]))

                # set timestamp
                os.utime(text_file, (year_timestamp, year_timestamp))

                if mode == "train_and_test":
                    (self.path / "test").mkdir(parents=True, exist_ok=True)
                    text_file = self.path / "test" / f"{year}.csv"
                    with open(text_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(year_rows["test"]))

                    # set timestamp
                    os.utime(text_file, (year_timestamp, year_timestamp))

            else:
                assert mode == "all"
                text_file = self.path / f"{year}.csv"
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(year_rows["all"]))

                # set timestamp
                os.utime(text_file, (year_timestamp, year_timestamp))

        if add_final_dummy_year:
            data_directories = (
                [self.path]
                if mode == "all"
                else ([self.path / "train"] + ([self.path / "test"] if mode == "train_and_test" else []))
            )
            for data_dir in data_directories:
                dummy_year = year + 1
                year_timestamp = create_timestamp(year=1970, month=1, day=dummy_year - 2011)
                text_file = data_dir / f"{dummy_year}.csv"
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(["dummy\t0"]))

                # set timestamp
                os.utime(text_file, (year_timestamp, year_timestamp))

        os.remove(os.path.join(self.path, "huffpost.pkl"))


if __name__ == "__main__":
    main()
