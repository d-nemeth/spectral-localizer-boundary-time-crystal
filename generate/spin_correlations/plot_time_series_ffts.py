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
        "font.size": 8,
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

# Select run to plot
RUN = "run_004"

base_dir = Path("supplemental_figures_datasets") / "spin_correlations_time_series_data"

run_dir = base_dir / RUN


figures_dir = run_dir / "figures"

figures_dir.mkdir(
    parents=True,
    exist_ok=True,
)

# Load data
data_file = run_dir / "data.pkl"

with open(
    data_file,
    "rb",
) as f:
    results = pickle.load(f)

# Load metadata
info_file = run_dir / "info.json"

with open(
    info_file,
    "r",
) as f:
    metadata = json.load(f)

# Parameters
Omega = metadata["Omega"]

N_list = metadata["N_list"]

Gamma_list = np.array(
    metadata["Gamma_list"],
    dtype=float,
)

T = metadata["T"]

n_steps = metadata["n_steps"]

data_raw = results["data"]

# Rebuild the time grid
times = np.linspace(
    0,
    T,
    n_steps,
)

# Choose dataset
N = 320

Gamma_idx = 0

Gamma = Gamma_list[Gamma_idx]

# Load data
if N in data_raw:
    data = data_raw[N]

elif str(N) in data_raw:
    data = data_raw[str(N)]

else:
    raise KeyError(f"Could not find data for N={N}")

# Load signals
sig1 = np.array(
    data["signals"]["sig1"][Gamma_idx],
    dtype=float,
)

sig2 = np.array(
    data["signals"]["sig2"][Gamma_idx],
    dtype=float,
)

sig3 = np.array(
    data["signals"]["sig3"][Gamma_idx],
    dtype=float,
)


# FFT helper
def compute_fft(
    times,
    signal,
    Omega,
):
    """
    Compute properly normalized single-sided FFT spectrum.

    Includes:
    - Correct single-sided normalization
    """

    signal = np.real(signal)

    dt = times[1] - times[0]

    fft_complex = np.fft.rfft(signal)

    freqs = np.fft.rfftfreq(
        len(signal),
        d=dt,
    )

    fft_vals = np.abs(fft_complex)
    fft_vals = 2 * fft_vals / len(signal)

    omegas = 2 * np.pi * freqs / Omega

    return (
        omegas,
        fft_vals,
    )


# Conmpute FFTs
omegas, fft1 = compute_fft(
    times,
    sig1,
    Omega,
)

_, fft2 = compute_fft(
    times,
    sig2,
    Omega,
)

_, fft3 = compute_fft(
    times,
    sig3,
    Omega,
)

# Cutoff
omega_max = 5.0

mask_cut = omegas <= omega_max

omegas_cut = omegas[mask_cut]

fft1_cut = fft1[mask_cut]

fft2_cut = fft2[mask_cut]

fft3_cut = fft3[mask_cut]

# Figure
fig, axs = plt.subplots(
    3,
    1,
    figsize=(3.4, 5.2),
    sharex=True,
)


fft_data = [
    (
        fft1_cut,
        r"$|\tilde{O}_1(\omega)|$",
        1,
    ),
    (
        fft2_cut,
        r"$|\tilde{O}_2(\omega)|$",
        2,
    ),
    (
        fft3_cut,
        r"$|\tilde{O}_3(\omega)|$",
        3,
    ),
]

panel_labels = [
    r"d)",
    r"e)",
    r"f)",
]


for ax, (
    fft_vals,
    ylabel,
    harmonic,
), panel_label in zip(
    axs,
    fft_data,
    panel_labels,
):
    ax.text(
        -0.08,
        0.95,
        panel_label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
    )

    ax.plot(
        omegas_cut,
        fft_vals,
        linewidth=0.9,
    )

    # Guide lines at harmonics
    for n in [
        1,
        2,
        3,
    ]:
        ax.axvline(
            n,
            linestyle="--",
            linewidth=0.7,
            color="black",
            alpha=0.35,
        )

    ax.axvline(
        harmonic,
        linestyle=":",
        linewidth=0.9,
        color="black",
        alpha=0.9,
    )

    ax.set_ylabel(ylabel)

    ax.tick_params(
        which="both",
        top=True,
        right=True,
    )
    formatter = ScalarFormatter(useMathText=True)

    formatter.set_powerlimits((-2, 2))

    ax.yaxis.set_major_formatter(formatter)

    ax.yaxis.get_offset_text().set_fontsize(6)

    ax.grid(False)

    ax.set_ylim(0)

    ax.yaxis.set_major_locator(MaxNLocator(3))


axs[-1].set_xlabel(r"$\omega/\Omega$")

axs[-1].set_xlim(
    0,
    4,
)


fig.tight_layout(
    pad=0.4,
    h_pad=0.3,
)

# Save
save_name = f"fft_spectra_N_{N}_Gamma_{Gamma:.2e}"

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
