import pandas as pd


def generate_real_training_end_timestamp(df_trainings: pd.DataFrame) -> pd.Series:
    """For sparse datasets we want to use next_training_start-1 as training
    interval end instead of last_timestamp as there could be a long gap between
    the max(sample_time) in one training batch and the min(sample_time) in the
    next training batch. e.g. if we want to train for 1.1.2020-31.12.2020 but
    only have timestamps on 1.1.2020, last_timestamp would be 1.1.2020, but the
    next training would start on 1.1.2021.

    Args:
        df_trainings: The pipeline stage execution tracking information including training and model infos.

    Returns:
        The real last timestamp series.
    """
    return df_trainings["first_timestamp"].shift(-1, fill_value=df_trainings.iloc[-1]["last_timestamp"] + 1) - 1
