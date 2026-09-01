# plot_S1_scaling.py

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    plt.style.use(["science"])
except Exception:
    pass

mpl.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "axes.linewidth": 0.8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "font.size": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "legend.frameon": False,
    }
)

# Select run to plot
RUN = "run_003"

N_list_plot = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]

base_dir = Path("supplemental_figures_datasets") / "spin_correlations_scaling_data"

run_dir = base_dir / RUN

figures_dir = run_dir / "figures"

figures_dir.mkdir(
    parents=True,
    exist_ok=True,
)

with open(run_dir / "data.pkl", "rb") as f:
    results = pickle.load(f)

with open(run_dir / "info.json", "r") as f:
    metadata = json.load(f)

data_raw = results["data"]

data = {}

for N in N_list_plot:
    if N in data_raw:
        data[N] = data_raw[N]

    elif str(N) in data_raw:
        data[N] = data_raw[str(N)]

observable_info = [
    ("A1", r"$k=1$", 1),
    ("A2", r"$k=2$", 2),
    ("A3", r"$k=3$", 3),
]

panel_labels = [
    r"a)",
    r"b)",
    r"c)",
]


def add_panel_label(
    ax,
    label,
):

    ax.text(
        -0.12,
        1.0,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
    )


# Figure setup
fig, axs = plt.subplots(
    1,
    3,
    figsize=(7.1, 2.2),
    sharex=True,
)

for ax, panel_label, (
    key,
    title,
    expected_k,
) in zip(
    axs,
    panel_labels,
    observable_info,
):
    add_panel_label(
        ax,
        panel_label,
    )

    for N in N_list_plot:
        Gamma_vals = np.array(
            data[N]["Gamma"],
            dtype=float,
        )

        A_vals = np.array(
            data[N][key],
            dtype=float,
        )

        ax.loglog(
            Gamma_vals,
            A_vals,
            "o-",
            linewidth=0.8,
            markersize=3,
            label=rf"$N={N}$",
        )

    # Guide line for expected power-law scaling
    N_ref = N_list_plot[-1]

    Gamma_ref = np.array(
        data[N_ref]["Gamma"],
        dtype=float,
    )

    A_ref = np.array(
        data[N_ref][key],
        dtype=float,
    )

    idx_ref = len(Gamma_ref) // 2

    guide_x = np.array(
        [
            Gamma_ref[0],
            Gamma_ref[-1],
        ]
    )

    guide_y = A_ref[idx_ref] * (guide_x / Gamma_ref[idx_ref]) ** expected_k

    ax.loglog(
        guide_x,
        guide_y,
        "--",
        linewidth=0.8,
        color="black",
        alpha=0.6,
    )

    # Legend for guide line
    guide_handle = mpl.lines.Line2D(
        [],
        [],
        linestyle="--",
        linewidth=0.8,
        color="black",
        alpha=0.6,
        label=rf"$\Gamma^{expected_k}$",
    )

    mini_legend = ax.legend(
        handles=[guide_handle],
        loc="upper left",
        fontsize=6,
        frameon=False,
        handlelength=1.2,
    )

    ax.add_artist(mini_legend)

    # Labels and axes
    ax.set_title(
        title,
        pad=3,
    )

    ax.set_xlabel(r"$\Gamma/\Omega$")

    ax.tick_params(
        which="both",
        top=True,
        right=True,
    )
    ax.set_xlim(min(Gamma_vals), max(Gamma_vals))


axs[0].set_ylabel(r"$A_k(N)$")

axs[-1].legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
)

fig.tight_layout(
    pad=0.4,
)

# Save
fig.savefig(
    figures_dir / "S1_scaling.pdf",
    bbox_inches="tight",
)

fig.savefig(
    figures_dir / "S1_scaling.png",
    bbox_inches="tight",
)

plt.show()
