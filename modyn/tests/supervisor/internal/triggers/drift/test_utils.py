import pandas as pd
import torch

from modyn.supervisor.internal.triggers.drift.utils import convert_tensor_to_df


def test_convert_tensor_to_df_without_prefix() -> None:
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    df = convert_tensor_to_df(tensor)
    expected_df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]]).astype("float64")
    pd.testing.assert_frame_equal(df, expected_df)


def test_convert_tensor_to_df_with_prefix() -> None:
    tensor = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    column_name_prefix = "feature_"
    df = convert_tensor_to_df(tensor, column_name_prefix=column_name_prefix)
    expected_df = pd.DataFrame([[5.0, 6.0], [7.0, 8.0]], columns=["feature_0", "feature_1"]).astype("float64")
    pd.testing.assert_frame_equal(df, expected_df)


def test_convert_tensor_to_df_on_gpu() -> None:
    if torch.cuda.is_available():
        tensor = torch.tensor([[13.0, 14.0], [15.0, 16.0]]).to("cuda")
        df = convert_tensor_to_df(tensor)
        expected_df = pd.DataFrame([[13.0, 14.0], [15.0, 16.0]]).astype("float64")
        pd.testing.assert_frame_equal(df, expected_df)
