# plot_S3_saturation.py

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter

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
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "font.size": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
    }
)

# Select run to plot
RUN = "run_003"

N_list_plot = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]

base_dir = Path("supplemental_figures_datasets") / "spin_correlations_scaling_data"

run_dir = base_dir / RUN

figures_dir = run_dir / "figures"

# Load data
with open(run_dir / "data.pkl", "rb") as f:
    results = pickle.load(f)

with open(run_dir / "info.json", "r") as f:
    metadata = json.load(f)

Omega = metadata["Omega"]

Gamma_list = np.array(
    metadata["Gamma_list"],
    dtype=float,
)

data_raw = results["data"]

data = {}

for N in N_list_plot:
    if N in data_raw:
        data[N] = data_raw[N]

    elif str(N) in data_raw:
        data[N] = data_raw[str(N)]


# Parameters for plotting
Gamma_idx = -1

Gamma0 = Gamma_list[Gamma_idx]

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


# Figure setup
fig, axs = plt.subplots(
    1,
    3,
    figsize=(7.1, 2.2),
)

for ax, panel_label, (
    key,
    title,
    k_rank,
) in zip(
    axs,
    panel_labels,
    observable_info,
):
    ax.text(
        -0.12,
        1.1,
        panel_label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
    )

    N_vals = np.array(
        N_list_plot,
        dtype=float,
    )

    A_vals = np.array(
        [data[N][key][Gamma_idx] for N in N_list_plot],
        dtype=float,
    )

    scaled_vals = A_vals / (Gamma0**k_rank)

    ax.plot(
        N_vals,
        scaled_vals,
        "o-",
        linewidth=0.9,
        markersize=3.5,
    )

    ax.set_title(
        title,
        pad=3,
    )

    ax.set_xlabel(r"$N$")

    ax.tick_params(
        which="both",
        top=True,
        right=True,
    )
    ax.set_xlim(40, 400)

    formatter = ScalarFormatter(useMathText=True)

    formatter.set_powerlimits((-2, 2))

    ax.yaxis.set_major_formatter(formatter)

    ax.yaxis.get_offset_text().set_fontsize(6)

    ax.yaxis.set_major_locator(MaxNLocator(3))


axs[0].set_ylabel(r"$A_k/\Gamma^k$")

fig.tight_layout(
    pad=0.4,
)


# Save
fig.savefig(
    figures_dir / "S3_saturation.pdf",
    bbox_inches="tight",
)

fig.savefig(
    figures_dir / "S3_saturation.png",
    bbox_inches="tight",
)

plt.show()
