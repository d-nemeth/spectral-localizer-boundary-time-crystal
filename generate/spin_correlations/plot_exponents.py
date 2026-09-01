# plot_S2_exponents.py

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

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
RUN = "run_002"

N_list_plot = [20, 40, 80, 160, 320]

base_dir = Path("supplemental_figures_datasets") / "spin_correlations_scaling_data"

run_dir = base_dir / RUN

figures_dir = run_dir / "figures"

# Load data
with open(run_dir / "data.pkl", "rb") as f:
    results = pickle.load(f)

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


def fit_power_law(
    Gamma_vals,
    A_vals,
):

    mask = np.isfinite(Gamma_vals) & np.isfinite(A_vals) & (A_vals > 1e-14)

    logG = np.log10(Gamma_vals[mask])

    logA = np.log10(A_vals[mask])

    slope, intercept, r_value, _, _ = linregress(
        logG,
        logA,
    )

    return slope


# Figure setup
fig, ax = plt.subplots(figsize=(3.35, 2.45))

for key, label, expected_k in observable_info:
    fitted_slopes = []

    for N in N_list_plot:
        Gamma_vals = np.array(
            data[N]["Gamma"],
            dtype=float,
        )

        A_vals = np.array(
            data[N][key],
            dtype=float,
        )

        slope = fit_power_law(
            Gamma_vals,
            A_vals,
        )

        fitted_slopes.append(slope)

    ax.plot(
        1
        / np.array(
            N_list_plot,
            dtype=float,
        ),
        fitted_slopes,
        "o-",
        linewidth=0.9,
        markersize=3.5,
        label=label,
    )

    ax.axhline(
        expected_k,
        linestyle="--",
        linewidth=0.7,
        color="black",
        alpha=0.45,
    )


# Labels and axes
ax.text(
    -0.16,
    1.05,
    r"a)",
    transform=ax.transAxes,
    fontsize=9,
    fontweight="bold",
)

ax.set_xlabel(r"$1/N$")

ax.set_ylabel(r"Fitted exponent $p_k(N)$")

ax.tick_params(
    which="both",
    top=True,
    right=True,
)

ax.legend(
    loc="best",
)

fig.tight_layout(
    pad=0.4,
)

# Save
fig.savefig(
    figures_dir / "S2_exponents.pdf",
    bbox_inches="tight",
)

fig.savefig(
    figures_dir / "S2_exponents.png",
    bbox_inches="tight",
)

plt.show()
