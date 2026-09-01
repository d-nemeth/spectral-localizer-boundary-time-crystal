from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (
    MaxNLocator,
    ScalarFormatter,
)

try:
    plt.style.use(["science"])

except Exception:
    pass

mpl.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.size": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
    }
)

RUN = "run_006"

base_dir = Path("supplemental_figures_datasets") / "spin_correlations_time_series_data"

run_dir = base_dir / RUN

figures_dir = run_dir / "figures"

figures_dir.mkdir(
    parents=True,
    exist_ok=True,
)

data_file = run_dir / "data.pkl"

with open(
    data_file,
    "rb",
) as f:
    results = pickle.load(f)

info_file = run_dir / "info.json"

with open(
    info_file,
    "r",
) as f:
    metadata = json.load(f)

Omega = metadata["Omega"]

N_list = metadata["N_list"]

Gamma_list = np.array(
    metadata["Gamma_list"],
    dtype=float,
)

T = metadata["T"]

n_steps = metadata["n_steps"]

times = np.linspace(
    0,
    T,
    n_steps,
)

data_raw = results["data"]

N_vals_plot = [
    10,
    400,
]

Gamma_indices_plot = [
    0,
    -1,
]

panel_labels = [
    r"a)",
    r"b)",
    r"c)",
    r"d)",
]

plot_cases = []


# Analytical function
def func(t, N, Gamma):
    F_0 = Gamma * (N + 2) / (3 * N)
    decay_rate = 3 * Gamma / (2 * N)
    omega_0 = np.sqrt(Omega**2 + (Gamma**2) / (2 * N**2))
    omega = np.sqrt(omega_0**2 - (decay_rate / 2) ** 2)
    mz_pi = -(Gamma * N * F_0) / (2 * N**2 * Omega**2 + Gamma**2)
    f_1 = mz_pi * (1 - np.exp(-decay_rate * t / 2) * (np.cos(omega * t)))
    f_2 = (
        -(F_0 + decay_rate / 2 * mz_pi)
        / omega
        * (np.exp(-decay_rate * t / 2) * np.sin(omega * t))
    )
    f_2 = -(F_0) / omega * (np.exp(-decay_rate * t / 2) * np.sin(omega * t))
    return f_1 + f_2


for N in N_vals_plot:
    if N in data_raw:
        data = data_raw[N]

    elif str(N) in data_raw:
        data = data_raw[str(N)]

    else:
        raise KeyError(f"Could not find data for N={N}")

    for Gamma_idx in Gamma_indices_plot:
        Gamma = Gamma_list[Gamma_idx]

        sig1 = np.array(
            data["signals"]["sig1"][Gamma_idx],
            dtype=float,
        )

        plot_cases.append(
            {
                "N": N,
                "Gamma": Gamma,
                "signal": sig1,
            }
        )

fig, axs = plt.subplots(
    2,
    2,
    figsize=(7.0, 4.8),
    sharex=True,
)

axs = axs.flatten()

for ax, case, panel_label in zip(
    axs,
    plot_cases,
    panel_labels,
):
    N = case["N"]

    Gamma = case["Gamma"]

    sig = case["signal"]

    ax.text(
        -0.10,
        0.95,
        panel_label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
    )

    analytical = np.array([func(t, N, Gamma) for t in times])

    xvals = Gamma * times

    ax.plot(
        xvals,
        analytical,
        linestyle="-",
        linewidth=1.6,
        color="blue",
        label="Analytical",
        zorder=3,
    )

    ax.plot(
        xvals[::20],
        sig[::20],
        linestyle="None",
        marker="o",
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.8,
        color="black",
        alpha=0.9,
        label="Numerical",
        zorder=4,
    )
    ax.set_title(
        rf"$N={N},\ \Gamma={Gamma:.1f}$",
        fontsize=10,
        pad=3,
    )

    ax.set_ylabel(r"$ m_z(t)$")

    ax.tick_params(
        which="both",
        top=True,
        right=True,
    )

    formatter = ScalarFormatter(useMathText=True)

    formatter.set_powerlimits((-2, 2))

    ax.yaxis.set_major_formatter(formatter)

    ax.yaxis.get_offset_text().set_fontsize(6)

    ax.yaxis.set_major_locator(MaxNLocator(3))

    ax.grid(False)

    ax.set_xlim(0, 10)
    ax.set_ylim(np.min(sig), np.max(sig))

for ax in axs[-2:]:
    ax.set_xlabel(r"$\Gamma t$")

fig.tight_layout(
    pad=0.4,
    h_pad=0.4,
    w_pad=0.5,
)

save_name = "time_series_comparison"

fig.savefig(
    figures_dir / f"{save_name}.pdf",
    bbox_inches="tight",
)

fig.savefig(
    figures_dir / f"{save_name}.png",
    bbox_inches="tight",
)

print(f"\nSaved figures to:\n{figures_dir}\n")

plt.show()
