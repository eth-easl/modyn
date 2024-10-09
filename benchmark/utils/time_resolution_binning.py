import os
from pathlib import Path

import pandas as pd

from modyn.const.types import TimeResolution

HELP_RESOLUTION = (
    "Resolution of the timestamps. "
    "If 'year', the timestamp will be the first day of the year. "
    "If 'month', the timestamp will be the first day of the month. "
    "If 'day', the timestamp will be the first hour of the day. "
    "If 'hour', the timestamp will be the first minute of the hour. "
    "If 'minute', the timestamp will be the first second of the minute. "
    "If 'second', the timestamp will be the exact second."
)


def bin_dataframe_wrt_time_resolution(
    df: pd.DataFrame, resolution: TimeResolution, datetime_col: str
) -> dict[pd.Period, pd.DataFrame]:
    """Splits the dataframe into groups according to the time resolution."""
    cleaned_df = df.copy()

    pandas_time_unit = {
        "year": "Y",
        "month": "M",
        "week": "W",
        "day": "D",
        "hour": "h",
        "minute": "min",
        "second": "s",
    }[resolution]

    # bin into groups according to resolution
    cleaned_df["datetime_bin"] = cleaned_df[datetime_col].dt.to_period(pandas_time_unit)

    # partition into groups
    partitions: dict[pd.Period, pd.DataFrame] = {}
    for name in sorted(cleaned_df["datetime_bin"].unique()):
        partitions[name] = cleaned_df[cleaned_df["datetime_bin"] == name]

    return partitions


def df_to_csv_with_timestamp(df: pd.DataFrame, period: pd.Period, data_dir: Path) -> None:
    """Stores the dataframe in a file with the timestamp."""
    label_file = data_dir / f"{str(period).replace('/', '_')}.csv"
    df.to_csv(label_file, index=False, sep="\t", lineterminator="\n", header=False)
    timestamp = int(period.to_timestamp().to_pydatetime().timestamp())
    os.utime(label_file, (timestamp, timestamp))
