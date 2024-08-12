import pandas as pd
import torch


def convert_tensor_to_df(t: torch.Tensor, column_name_prefix: str | None = None) -> pd.DataFrame:
    matrix_numpy = t.cpu().detach().numpy()
    df = pd.DataFrame(matrix_numpy).astype("float64")
    if column_name_prefix is not None:
        df.columns = [column_name_prefix + str(x) for x in df.columns]
    return df
