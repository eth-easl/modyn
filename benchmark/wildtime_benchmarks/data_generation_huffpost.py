import os
import pickle
from pathlib import Path
from typing import Annotated

import torch
import typer
from benchmark_utils import create_timestamp, download_if_not_exists
from modyn.utils.logging import setup_logging
from torch.utils.data import Dataset
from tqdm import tqdm

logger = setup_logging(__name__)


def main(
    dir: Annotated[Path, typer.Argument(help="Path to Yearbook data directory")],
    dummy_year: Annotated[
        bool,
        typer.Option(help="Add a final dummy year to train also on the last trigger in Modyn"),
    ] = False,
    all: Annotated[
        bool,
        typer.Option(help="Store all the available data, including the validation and test sets."),
    ] = False,
) -> None:
    """Huffpost data generation script."""
    logger.info(f"Downloading data to {dir}")

    HuffpostDownloader(dir).store_data(all, dummy_year)


class HuffpostDownloader(Dataset):
    time_steps = [i for i in range(2012, 2019)]
    input_dim = 44
    num_classes = 11
    drive_id = "1jKqbfPx69EPK_fjgU9RLuExToUg7rwIY"
    file_name = "huffpost.pkl"

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        return (
            self._dataset["title"][idx],
            torch.LongTensor([self._dataset["category"][idx]])[0],
        )

    def __len__(self) -> int:
        return len(self._dataset["category"])

    def __init__(self, data_dir: Path):
        super().__init__()

        download_if_not_exists(
            drive_id=self.drive_id,
            destination_dir=data_dir,
            destination_file_name=self.file_name,
        )
        with open(os.path.join(data_dir, self.file_name), "rb") as f:
            datasets = pickle.load(f)
        assert self.time_steps == list(sorted(datasets.keys()))
        self._dataset = datasets
        self.path = data_dir

    def store_data(self, store_all_data: bool, add_final_dummy_year: bool) -> None:
        for year in tqdm(self._dataset):
            year_timestamp = create_timestamp(year=1970, month=1, day=year - 2011)
            year_rows = []
            splits = [0, 1] if store_all_data else [0]
            for split in splits:
                for i in range(len(self._dataset[year][split]["headline"])):
                    text = self._dataset[year][split]["headline"][i]
                    label = self._dataset[year][split]["category"][i]
                    csv_row = f"{text}\t{label}"
                    year_rows.append(csv_row)

            # store the sentences
            text_file = os.path.join(self.path, f"{year}.csv")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write("\n".join(year_rows))

            # set timestamp
            os.utime(text_file, (year_timestamp, year_timestamp))

        if add_final_dummy_year:
            dummy_year = year + 1
            year_timestamp = create_timestamp(year=1970, month=1, day=dummy_year - 2011)
            text_file = os.path.join(self.path, f"{dummy_year}.csv")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write("\n".join(["dummy\t0"]))

            # set timestamp
            os.utime(text_file, (year_timestamp, year_timestamp))

        os.remove(os.path.join(self.path, "huffpost.pkl"))


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
