import os
from pathlib import Path
from typing import Annotated

import typer
from modyn.utils.logging import setup_logging

logger = setup_logging(__name__)

DAY_LENGTH_SECONDS = 24 * 60 * 60


def main(dir: Annotated[Path, typer.Argument(help="Path to Criteo data directory")]) -> None:
    "Criteo Timestamp script"
    validate_dir(dir)
    fix_timestamps(dir)


def validate_dir(path: Path) -> None:
    for i in range(0, 24):
        subpath = path / f"day{i}"
        if not subpath.exists():
            raise ValueError(f"Did not find directory for day {i} (checked {subpath.resolve()})")


def fix_timestamps(path: Path) -> None:
    for i in range(0, 24):
        subpath = path / f"day{i}"
        fix_day(subpath, i)


def fix_day(path: Path, day: int) -> None:
    assert day >= 0 and day < 24
    timestamp = (day * DAY_LENGTH_SECONDS) + 1  # avoid off by ones in storage by adding + 1

    filelist = path.glob("**/*.bin")
    for file in filelist:
        os.utime(file, (timestamp, timestamp))


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
