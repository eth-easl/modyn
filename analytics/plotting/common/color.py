from typing import Any

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def get_rdbu_wo_white(
    palette: str = "RdBu",
    strip: tuple[float, float] | None = (0.35, 0.65),
    nvalues: int = 100,
) -> LinearSegmentedColormap | str:
    if strip is None:
        return palette

    # Truncate the "RdBu" colormap to exclude the light colors
    rd_bu_cmap = plt.get_cmap(palette)
    custom_cmap_blu = rd_bu_cmap(np.linspace(0.0, strip[0], nvalues // 2))
    custom_cmap_red = rd_bu_cmap(np.linspace(strip[1], 1.0, nvalues // 2))
    cmap = LinearSegmentedColormap.from_list("truncated", np.concatenate([custom_cmap_blu, custom_cmap_red]))
    return cmap


def gen_categorical_map(categories: list) -> dict[Any, tuple[float, float, float]]:
    palette = (
        sns.color_palette("bright")
        + sns.color_palette("dark")
        + sns.color_palette("colorblind")
        + sns.color_palette("pastel")
        + sns.color_palette("Paired") * 100
    )[: len(categories)]
    color_map = dict(zip(categories, palette))
    return color_map


def discrete_colors(n: int = 10):
    return sns.color_palette("RdBu", n)


def discrete_color(i: int, n: int = 10) -> tuple[float, float, float]:
    palette = discrete_colors(n)
    return palette[i % n]


def main_colors(light: bool = False) -> list[tuple[float, float, float]]:
    rdbu_palette = discrete_colors(10)
    colorblind_palette = sns.color_palette("colorblind", 10)

    if light:
        return [
            rdbu_palette[-2],
            rdbu_palette[2],
            colorblind_palette[-2],
            colorblind_palette[1],
            colorblind_palette[2],
            colorblind_palette[3],
            colorblind_palette[4],
        ]
    return [
        rdbu_palette[-1],
        rdbu_palette[1],
        colorblind_palette[-2],
        colorblind_palette[1],
        colorblind_palette[2],
        colorblind_palette[3],
        colorblind_palette[4],
    ]


def main_color(i: int, light: bool = False) -> tuple[float, float, float]:
    return main_colors(light=light)[i]
