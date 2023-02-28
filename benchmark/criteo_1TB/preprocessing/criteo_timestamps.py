import argparse
import logging
import os
import pathlib

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

DAY_LENGTH_SECONDS = 24 * 60 * 60


def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description="Criteo Timestamp script")
    parser_.add_argument("dir", type=pathlib.Path, action="store", help="Path to Criteo data directory")

    return parser_


def main():
    parser = setup_argparser()
    args = parser.parse_args()

    validate_dir(args.dir)
    fix_timestamps(args.dir)


def validate_dir(path: pathlib.Path):
    for i in range(0, 24):
        subpath = path / f"day{i}"
        if not subpath.exists():
            raise ValueError(f"Did not find directory for day {i} (checked {subpath.resolve()})")


def fix_timestamps(path: pathlib.Path):
    for i in range(0, 24):
        subpath = path / f"day{i}"
        fix_day(subpath, i)


def fix_day(path: pathlib.Path, day: int):
    assert day >= 0 and day < 24
    timestamp = (day * DAY_LENGTH_SECONDS) + 1  # avoid off by ones in storage by adding + 1

    filelist = path.glob("**/*.bin")
    for file in filelist:
        os.utime(file, (timestamp, timestamp))


if __name__ == "__main__":
    main()
