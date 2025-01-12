import matplotlib.font_manager as fm
from matplotlib import pyplot as plt

__loaded = False


def load_font() -> None:
    global __loaded
    if __loaded:
        return

    cmu_fonts = [
        x
        for x in fm.findSystemFonts(fontpaths=["/Users/robinholzinger/Library/Fonts/"])
        if "cmu" in x or "p052" in x.lower()
    ]

    for font in cmu_fonts:
        # Register the font with Matplotlib's font manager
        # font_prop = fm.FontProperties(fname=font)
        fm.fontManager.addfont(font)

    assert len([f.name for f in fm.fontManager.ttflist if "cmu" in f.name.lower()]) >= 2


def setup_font(small_label: bool = False, small_title: bool | None = None, small_ticks: bool = True) -> None:
    load_font()
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "P052"  # latex default: "CMU Serif", robin thesis: P052
    plt.rcParams["legend.fontsize"] = "small" if small_ticks else "medium"
    plt.rcParams["xtick.labelsize"] = "small" if small_ticks else "medium"
    plt.rcParams["ytick.labelsize"] = "small" if small_ticks else "medium"
    plt.rcParams["axes.labelsize"] = "small" if small_label else "medium"
    if small_title is not None:
        plt.rcParams["axes.titlesize"] = "small" if small_title else "medium"
