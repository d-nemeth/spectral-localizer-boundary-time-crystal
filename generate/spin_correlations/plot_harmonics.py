# plot_S4_fft.py

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
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "font.size": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


# Select run to plot
RUN = "run_002"

N_fft = 320

Gamma_idx_fft = -1

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

T = metadata["T"]

n_steps = metadata["n_steps"]

times = np.linspace(
    0,
    T,
    n_steps,
)

data_raw = results["data"]

if N_fft in data_raw:
    data = data_raw[N_fft]
else:
    data = data_raw[str(N_fft)]

# Load actual peak locations
omega1_peak = data["omega1_over_Omega"][Gamma_idx_fft]

omega2_peak = data["omega2_over_Omega"][Gamma_idx_fft]

omega3_peak = data["omega3_over_Omega"][Gamma_idx_fft]

actual_peak_locations = [
    omega1_peak,
    omega2_peak,
    omega3_peak,
]

# Load signals
sig1 = np.array(
    data["signals"]["sig1"][Gamma_idx_fft],
    dtype=float,
)

sig2 = np.array(
    data["signals"]["sig2"][Gamma_idx_fft],
    dtype=float,
)

sig3 = np.array(
    data["signals"]["sig3"][Gamma_idx_fft],
    dtype=float,
)

signals = [
    sig1,
    sig2,
    sig3,
]


# FFT helper function
def compute_fft(
    signal,
):
    """
    Properly normalized FFT using:

    - DC subtraction
    - real FFT
    """

    # Remove DC component
    signal = signal - np.mean(signal)

    dt = times[1] - times[0]

    fft_complex = np.fft.rfft(signal)

    freqs = np.fft.rfftfreq(
        len(signal),
        d=dt,
    )

    fft_vals = 2 * np.abs(fft_complex) / len(signal)

    omegas = 2 * np.pi * freqs / Omega

    return (
        omegas,
        fft_vals,
    )


fig, axs = plt.subplots(
    1,
    3,
    figsize=(7.1, 2.2),
)

titles = [
    r"$k=1$",
    r"$k=2$",
    r"$k=3$",
]

panel_labels = [
    r"a)",
    r"b)",
    r"c)",
]


for ax, panel_label, sig, title, peak_loc in zip(
    axs,
    panel_labels,
    signals,
    titles,
    actual_peak_locations,
):
    ax.text(
        -0.18,
        1.05,
        panel_label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
    )

    omegas, fft_vals = compute_fft(sig)

    ax.plot(
        omegas,
        fft_vals,
        linewidth=0.8,
    )

    ax.axvline(
        peak_loc,
        linestyle="--",
        linewidth=0.9,
        color="black",
        alpha=0.7,
    )

    ax.text(
        0.60,
        0.86,
        rf"$\omega/\Omega={peak_loc:.3f}$",
        transform=ax.transAxes,
        fontsize=6,
    )

    ax.set_title(
        title,
        pad=3,
    )

    ax.set_xlim(
        0,
        5,
    )

    ax.set_xlabel(r"$\omega/\Omega$")

    ax.tick_params(
        which="both",
        top=True,
        right=True,
    )


axs[0].set_ylabel("FFT amplitude")


axs[0].text(
    0.05,
    0.92,
    rf"$N={N_fft}$"
    "\n"
    rf"$\Gamma={Gamma_list[Gamma_idx_fft]:.2e}$",
    transform=axs[0].transAxes,
    fontsize=7,
    va="top",
)

fig.tight_layout(
    pad=0.4,
)

# Save
fig.savefig(
    figures_dir / "S4_fft_actual_peaks.pdf",
    bbox_inches="tight",
)

fig.savefig(
    figures_dir / "S4_fft_actual_peaks.png",
    bbox_inches="tight",
)

plt.show()
