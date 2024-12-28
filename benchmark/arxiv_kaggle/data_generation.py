import json
import shutil
from pathlib import Path
from typing import Annotated, Any

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


class ArxivKaggleDataGenerator:
    test_holdout = 0.25
    fields_to_keep = ["title", "category"]

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
        return self.data_dir / "arxiv-metadata-oai-snapshot.json"

    def clean_folder(self) -> None:
        self.data_json.unlink()

    def load_into_dataframe(self, keep_true_category: bool = False) -> pd.DataFrame:
        records = []
        for line in self.data_json.read_text().splitlines():
            record = json.loads(line)
            records.append({field: record[field] for field in ["title", "categories", "versions", "update_date"]})

        df = pd.DataFrame(records)
        return ArxivKaggleDataGenerator.sanitize_dataframe(df, keep_true_category=keep_true_category)

    def store_data(
        self, cleaned_df: pd.DataFrame, resolution: TimeResolution, test_split: bool, dummy_period: bool = False
    ) -> None:
        partitions = bin_dataframe_wrt_time_resolution(cleaned_df, resolution, "first_version_timestamp")

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
        def extract_first_version_timestamp(row: Any) -> Any:
            versions = row["versions"]
            if len(versions) == 0:
                return None
            return versions[0]["created"]

        transformed = raw_df

        # replace newlines and whitespace in title with spaces
        transformed["title"] = transformed["title"].str.replace("\n", " ").replace(r"\s+", " ", regex=True)

        # we only take the first category (like in the wilds)
        transformed["category"] = transformed["categories"].str.split(" ").str[0]

        if not keep_true_category:
            # to int-categorical
            transformed["category"] = pd.Categorical(transformed["category"]).codes

        # we only take the first version timestamp
        transformed["first_version_timestamp"] = transformed.apply(extract_first_version_timestamp, axis=1)
        transformed["first_version_timestamp"] = pd.to_datetime(transformed["first_version_timestamp"])

        # currently unused but we keep to be able to switch implementation easily
        transformed["update_date"] = pd.to_datetime(transformed["update_date"])

        return transformed[["title", "category", "first_version_timestamp", "update_date"]]


def main(
    dir: Annotated[Path, typer.Argument(help="Path to arxiv data directory")],
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
    """Arxiv data generation script."""

    downloader = ArxivKaggleDataGenerator(dir, raw_data)
    if not skip_extraction:
        downloader.extract_data(raw_data)
    df = downloader.load_into_dataframe()
    downloader.store_data(df, resolution, test_split, dummy_period)
    downloader.clean_folder()


if __name__ == "__main__":
    typer.run(main)
