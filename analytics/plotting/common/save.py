from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure


def save_plot(fig: Figure, name: str) -> None:
    for img_type in ["png", "svg", "pdf"]:
        img_path = Path(".data/_plots") / f"{name}.{img_type}"
        img_path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(img_path, bbox_inches="tight", transparent=True)


def save_csv_df(df: pd.DataFrame, name: str) -> None:
    csv_path = Path(".data/csv") / f"{name}.csv"
    csv_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(csv_path, index=False)
