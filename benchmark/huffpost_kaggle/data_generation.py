import json
import shutil
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from benchmark.utils.time_resolution_binning import (
    HELP_RESOLUTION,
    bin_dataframe_wrt_time_resolution,
    df_to_csv_with_timestamp,
)
from modyn.const.types import TimeResolution


class HuffpostKaggleDataGenerator:
    fields_to_keep = ["headline", "category"]
    test_holdout = 0.25

    def __init__(self, data_dir: Path, data_zip: Path) -> None:
        self.assert_data_exists(data_zip)
        self.data_dir = data_dir

    def assert_data_exists(self, data_zip: Path) -> None:
        assert data_zip.exists(), "Kaggle dataset zip file not found."

    def extract_data(self, data_zip: Path) -> None:
        shutil.unpack_archive(data_zip, self.data_dir)

    @property
    def data_json(self) -> Path:
        # Will be extracted automatically from the user provided raw zip file
        return self.data_dir / "News_Category_Dataset_v3.json"

    def clean_folder(self) -> None:
        self.data_json.unlink()

    def load_into_dataframe(self, keep_true_category: bool = False) -> pd.DataFrame:
        records = []
        for line in self.data_json.read_text().splitlines():
            record = json.loads(line)
            records.append({field: record[field] for field in ["headline", "category", "date"]})

        df = pd.DataFrame(records)
        return HuffpostKaggleDataGenerator.sanitize_dataframe(df, keep_true_category=keep_true_category)

    def store_data(
        self, cleaned_df: pd.DataFrame, resolution: TimeResolution, test_split: bool, dummy_period: bool = False
    ) -> None:
        partitions = bin_dataframe_wrt_time_resolution(cleaned_df, resolution, "date")

        if test_split:
            (self.data_dir / "train").mkdir(exist_ok=True, parents=True)
            (self.data_dir / "test").mkdir(exist_ok=True, parents=True)

        # store partitions in files
        for name, partition in tqdm(partitions.items()):
            if test_split:
                if partition.shape[0] <= 1:
                    # we need at least 2 samples to split
                    df_to_csv_with_timestamp(partition[self.fields_to_keep], name, self.data_dir / "train")
                else:
                    df_train, df_test = train_test_split(
                        partition[self.fields_to_keep],
                        test_size=self.test_holdout,
                        random_state=42,
                    )
                    df_to_csv_with_timestamp(df_train, name, self.data_dir / "train")
                    df_to_csv_with_timestamp(df_test, name, self.data_dir / "test")
            else:
                df_to_csv_with_timestamp(partition[self.fields_to_keep], name, self.data_dir)

        if dummy_period:
            df_to_csv_with_timestamp(
                df=pd.DataFrame([{"title": "dummy", "category": 1}]),
                period=max(partitions.keys()) + 1,
                data_dir=self.data_dir / "train" if test_split else self.data_dir,
            )

    @staticmethod
    def sanitize_dataframe(raw_df: pd.DataFrame, keep_true_category: bool = False) -> pd.DataFrame:
        transformed = raw_df

        # escape new lines
        transformed["headline"] = transformed["headline"].str.replace("\n", " ").replace(r"\s+", " ", regex=True)

        if not keep_true_category:
            # to int-categorical
            transformed["category"] = pd.Categorical(transformed["category"]).codes

        # parse the date
        transformed["date"] = pd.to_datetime(transformed["date"])

        return transformed[["headline", "category", "date"]]


def main(
    dir: Annotated[Path, typer.Argument(help="Path to huffpost data directory")],
    raw_data: Annotated[Path, typer.Argument(help="Path to raw Kaggle data zip file")],
    resolution: Annotated[
        TimeResolution,
        typer.Option(help=HELP_RESOLUTION),
    ],
    test_split: Annotated[
        bool,
        typer.Option(help="Split the data into train and test sets"),
    ] = True,
    dummy_period: Annotated[
        bool,
        typer.Option(help="Add a final dummy period to train also on the last trigger in Modyn"),
    ] = False,
    skip_extraction: Annotated[
        bool,
        typer.Option(
            help="Skip extraction of the raw data",
        ),
    ] = False,
) -> None:
    """Huffpost data generation script."""

    downloader = HuffpostKaggleDataGenerator(dir, raw_data)
    if not skip_extraction:
        downloader.extract_data(raw_data)
    df = downloader.load_into_dataframe()
    downloader.store_data(df, resolution, test_split, dummy_period)
    downloader.clean_folder()


if __name__ == "__main__":
    typer.run(main)
