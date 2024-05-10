import csv
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from benchmark_utils import create_timestamp, download_if_not_exists
from modyn.utils.logging import setup_logging
from torch.utils.data import Dataset
from tqdm import tqdm
from wilds import get_dataset

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
    daily: Annotated[
        bool,
        typer.Option(
            help=(
                "If specified, data is stored with real timestamps (dd/mm/yy). "
                "Otherwise, only the year is considered (as done in the other datasets)"
            )
        ),
    ] = False,
) -> None:
    """Yearbook data generation script."""

    logger.info(f"Downloading data to {dir}")

    downloader = FMOWDownloader(dir)
    downloader.store_data(daily, all, dummy_year)
    downloader.clean_folder()


class FMOWDownloader(Dataset):
    time_steps = list(range(16))
    input_dim = (3, 224, 224)
    num_classes = 62
    drive_id = "1s_xtf2M5EC7vIFhNv_OulxZkNvrVwIm3"
    file_name = "fmow.pkl"

    def __init__(self, data_dir: Path) -> None:
        download_if_not_exists(
            drive_id=self.drive_id,
            destination_dir=data_dir,
            destination_file_name=self.file_name,
        )
        with open(data_dir / self.file_name, "rb") as f:
            datasets = pickle.load(f)
        self._dataset = datasets
        try:
            self._root = get_dataset(dataset="fmow", root_dir=data_dir, download=True).root
        except ValueError:
            pass
        self.metadata = self.parse_metadata(data_dir)
        self.data_dir = data_dir

    def clean_folder(self) -> None:
        folder_path = self.data_dir / "fmow_v1.1"
        if folder_path.exists():
            shutil.rmtree(folder_path)

    def move_file_and_rename(self, index: int) -> None:
        source_dir = self.data_dir / "fmow_v1.1" / "images"
        if source_dir.exists() and source_dir.is_dir():
            src_file = os.path.join(source_dir, f"rgb_img_{index}.png")
            dest_file = os.path.join(self.data_dir, f"rgb_img_{index}.png")
            shutil.move(src_file, dest_file)
            new_name = os.path.join(self.data_dir, f"{index}.png")
            os.rename(dest_file, new_name)

    def store_data(self, store_daily: bool, store_all_data: bool, add_final_dummy_year: bool) -> None:

        for year in tqdm(self._dataset):
            splits = [0, 1] if store_all_data else [0]
            for split in splits:
                for i in range(len(self._dataset[year][split]["image_idxs"])):
                    index = self._dataset[year][split]["image_idxs"][i]
                    label = self._dataset[year][split]["labels"][i]

                    if store_daily:
                        raw_timestamp = self.metadata[index]["timestamp"]

                        if len(raw_timestamp) == 24:
                            timestamp = datetime.strptime(raw_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
                        else:
                            timestamp = datetime.strptime(raw_timestamp, "%Y-%m-%dT%H:%M:%SZ").timestamp()
                    else:
                        timestamp = create_timestamp(year=1970, month=1, day=year + 1)

                    # save label
                    label_file = os.path.join(self.data_dir, f"{index}.label")
                    with open(label_file, "w", encoding="utf-8") as f:
                        f.write(str(int(label)))
                    os.utime(label_file, (timestamp, timestamp))

                    # set image timestamp
                    self.move_file_and_rename(index)
                    image_file = os.path.join(self.data_dir, f"{index}.png")
                    os.utime(image_file, (timestamp, timestamp))

        if add_final_dummy_year:
            dummy_year = year + 1
            timestamp = create_timestamp(year=1970, month=1, day=dummy_year + 1)
            dummy_index = 1000000  # not used by any real sample (last: 99999)

            to_copy_image_file = os.path.join(self.data_dir, f"{index}.png")
            dummy_image_file = os.path.join(self.data_dir, f"{dummy_index}.png")
            shutil.copy(to_copy_image_file, dummy_image_file)
            os.utime(dummy_image_file, (timestamp, timestamp))

            to_copy_label_file = os.path.join(self.data_dir, f"{index}.label")
            dummy_label_file = os.path.join(self.data_dir, f"{dummy_index}.label")
            shutil.copy(to_copy_label_file, dummy_label_file)
            os.utime(dummy_label_file, (timestamp, timestamp))

    @staticmethod
    def parse_metadata(data_dir: Path) -> list:
        filename = data_dir / "fmow_v1.1" / "rgb_metadata.csv"
        metadata = []

        with open(filename, "r") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                picture_info = {"split": row[0], "timestamp": row[11]}
                metadata.append(picture_info)
        return metadata


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
