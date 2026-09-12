import matplotlib

def apply():
    matplotlib.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "none",
        "legend.framealpha": 0.9,
        "legend.fontsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })

PALETTE = {
    "primary": "#1F4E78",
    "accent": "#B22222",
    "teal": "#0072B2",
    "orange": "#D95F02",
    "green": "#009E73",
    "muted": "#7F7F7F",
    "grid": "#E0E0E0",
}

DATASET_COLORS = {
    "smd": "#0072B2",
    "smap": "#009E73",
    "cicids2017": "#D95F02",
    "unsw_nb15": "#CC79A7",
}
