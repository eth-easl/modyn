from datetime import datetime, timezone
from pathlib import Path

import gdown

DAY_LENGTH_SECONDS = 24 * 60 * 60


def download_if_not_exists(drive_id: str, destination_dir: Path, destination_file_name: str) -> None:
    """
    Function to download data from Google Drive. Used for Wild-time based benchmarks.
    This function is adapted from wild-time-data's maybe_download
    """
    destination_dir = Path(destination_dir)
    destination = destination_dir / destination_file_name
    if destination.exists():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    gdown.download(
        url=f"https://drive.google.com/u/0/uc?id={drive_id}&export=download&confirm=pbef",
        output=str(destination),
        quiet=False,
    )


def create_fake_timestamp(year: int, base_year: int) -> int:
    return ((year - base_year) * DAY_LENGTH_SECONDS) + 1


def create_timestamp(year: int, month: int = 1, day: int = 1) -> int:
    return int(datetime(year=year, month=month, day=day, tzinfo=timezone.utc).timestamp())
