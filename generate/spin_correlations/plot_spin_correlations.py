from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

##############################################################################
# STYLE
##############################################################################

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
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "legend.frameon": False,
    }
)


##############################################################################
# SELECT RUN
##############################################################################

RUN = "run_002"

base_dir = Path("supplemental_figures_datasets") / "spin_correlations_scaling_data"

run_dir = base_dir / RUN

figures_dir = run_dir / "figures"

figures_dir.mkdir(
    parents=True,
    exist_ok=True,
)


##############################################################################
# LOAD DATA
##############################################################################

with open(run_dir / "data.pkl", "rb") as f:
    results = pickle.load(f)

with open(run_dir / "info.json", "r") as f:
    metadata = json.load(f)


##############################################################################
# PARAMETERS
##############################################################################

Omega = metadata["Omega"]

N_list = metadata["N_list"]

Gamma_list = np.array(
    metadata["Gamma_list"],
    dtype=float,
)

T = metadata["T"]

n_steps = metadata["n_steps"]

save_raw_signals = metadata["save_raw_signals"]

data_raw = results["data"]

# Pickle may preserve integer keys, but JSON metadata uses normal ints.
# This makes access robust.
data = {}

for N in N_list:
    if N in data_raw:
        data[N] = data_raw[N]
    elif str(N) in data_raw:
        data[N] = data_raw[str(N)]
    else:
        raise KeyError(f"Could not find data for N={N}")


##############################################################################
# HELPERS
##############################################################################

observable_info = [
    ("A1", r"$k=1$", 1),
    ("A2", r"$k=2$", 2),
    ("A3", r"$k=3$", 3),
]


def add_panel_label(
    ax,
    label,
    x=-0.18,
    y=1.06,
):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(
    fig,
    name,
):
    pdf_path = figures_dir / f"{name}.pdf"
    png_path = figures_dir / f"{name}.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        bbox_inches="tight",
    )

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def fit_power_law(
    Gamma_vals,
    A_vals,
):
    Gamma_vals = np.array(
        Gamma_vals,
        dtype=float,
    )

    A_vals = np.array(
        A_vals,
        dtype=float,
    )

    mask = (
        np.isfinite(Gamma_vals)
        & np.isfinite(A_vals)
        & (Gamma_vals > 0)
        & (A_vals > 1e-14)
    )

    logG = np.log10(Gamma_vals[mask])

    logA = np.log10(A_vals[mask])

    slope, intercept, r_value, _, _ = linregress(
        logG,
        logA,
    )

    return slope, intercept, r_value**2


##############################################################################
# PRINT METADATA
##############################################################################

print("\nLoaded BTC scaling dataset")
print("-" * 50)
print(f"Run       = {RUN}")
print(f"Omega     = {Omega}")
print(f"N values  = {N_list}")
print(f"Gammas    = {Gamma_list}")
print(f"T         = {T}")
print(f"n_steps   = {n_steps}")


##############################################################################
# FIGURE S1 — SCALING OF HARMONIC AMPLITUDES
##############################################################################

fig, axs = plt.subplots(
    1,
    3,
    figsize=(7.1, 2.15),
    sharex=True,
)

panel_labels = [
    r"(a)",
    r"(b)",
    r"(c)",
]

for ax, panel_label, (key, title, expected_k) in zip(
    axs,
    panel_labels,
    observable_info,
):
    add_panel_label(
        ax,
        panel_label,
    )

    for N in N_list:
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
            markersize=3.0,
            linewidth=0.8,
            label=rf"$N={N}$",
        )

    # Guide line anchored to largest-N data.
    N_ref = N_list[-1]

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
        color="black",
        linewidth=0.8,
        alpha=0.65,
    )

    ax.text(
        0.06,
        0.08,
        rf"$\sim \Gamma^{expected_k}$",
        transform=ax.transAxes,
        fontsize=7,
    )

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

    ax.grid(False)

axs[0].set_ylabel(r"$A_k(N)$")

axs[-1].legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    handlelength=1.3,
)

fig.tight_layout(
    pad=0.4,
    w_pad=0.4,
)

save_figure(fig, "S_spin_correlations_scaling")


##############################################################################
# FIGURE S2 — FITTED EXPONENTS
##############################################################################

fig, ax = plt.subplots(figsize=(3.35, 2.45))

for key, label, expected_k in observable_info:
    fitted_slopes = []

    for N in N_list:
        Gamma_vals = np.array(
            data[N]["Gamma"],
            dtype=float,
        )

        A_vals = np.array(
            data[N][key],
            dtype=float,
        )

        slope, intercept, r2 = fit_power_law(
            Gamma_vals,
            A_vals,
        )

        fitted_slopes.append(slope)

        print(f"{key}, N={N}: slope = {slope:.4f}, R^2 = {r2:.6f}")

    ax.plot(
        1 / np.array(N_list, dtype=float),
        fitted_slopes,
        "o-",
        markersize=3.5,
        linewidth=0.9,
        label=label,
    )

    ax.axhline(
        expected_k,
        linestyle="--",
        linewidth=0.7,
        color="black",
        alpha=0.45,
    )

add_panel_label(
    ax,
    r"(a)",
    x=-0.16,
    y=1.05,
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

save_figure(fig, "S_fitted_exponents")


##############################################################################
# FIGURE S3 — LARGE-N RESCALED AMPLITUDES
##############################################################################

Gamma_idx = -1

Gamma0 = Gamma_list[Gamma_idx]

fig, axs = plt.subplots(
    1,
    3,
    figsize=(7.1, 2.15),
)

for ax, panel_label, (key, title, k_rank) in zip(
    axs,
    panel_labels,
    observable_info,
):
    add_panel_label(
        ax,
        panel_label,
    )

    N_vals = np.array(
        N_list,
        dtype=float,
    )

    A_vals = np.array(
        [data[N][key][Gamma_idx] for N in N_list],
        dtype=float,
    )

    scaled_vals = A_vals / (Gamma0**k_rank)

    ax.plot(
        N_vals,
        scaled_vals,
        "o-",
        markersize=3.5,
        linewidth=0.9,
    )

    ax.axhline(
        scaled_vals[-1],
        linestyle="--",
        linewidth=0.7,
        color="black",
        alpha=0.5,
    )

    if k_rank == 1:
        analytic_val = 1 / (3 * Omega)

        ax.axhline(
            analytic_val,
            linestyle=":",
            linewidth=0.9,
            color="black",
            alpha=0.8,
            label=r"$1/(3\Omega)$",
        )

        ax.legend(
            fontsize=6,
            loc="best",
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

axs[0].set_ylabel(r"$A_k(N,\Gamma)/\Gamma^k$")

axs[0].text(
    0.05,
    0.08,
    rf"$\Gamma={Gamma0:.2e}$",
    transform=axs[0].transAxes,
    fontsize=7,
)

fig.tight_layout(
    pad=0.4,
    w_pad=0.5,
)

save_figure(fig, "S_rescaled_large_N")


##############################################################################
# OPTIONAL FIGURE S4 — FOURIER SPECTRA
##############################################################################

if save_raw_signals:
    times = np.linspace(
        0,
        T,
        n_steps,
    )

    N_fft = N_list[-2]

    Gamma_idx_fft = min(
        5,
        len(Gamma_list) - 1,
    )

    Gamma_fft = Gamma_list[Gamma_idx_fft]

    sigs = [
        np.array(
            data[N_fft]["signals"][f"sig{k}"][Gamma_idx_fft],
            dtype=float,
        )
        for k in [1, 2, 3]
    ]

    dt = times[1] - times[0]

    freqs = np.fft.fftfreq(
        len(times),
        d=dt,
    )

    mask = freqs > 0

    omegas = 2 * np.pi * freqs[mask] / Omega

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(7.1, 2.15),
    )

    for ax, panel_label, sig, (_, title, _) in zip(
        axs,
        panel_labels,
        sigs,
        observable_info,
    ):
        add_panel_label(
            ax,
            panel_label,
        )

        sig = sig - np.mean(sig)

        fft_vals = np.abs(np.fft.fft(sig))

        fft_vals = 2 * fft_vals[mask] / len(times)

        ax.plot(
            omegas,
            fft_vals,
            linewidth=0.8,
        )

        for n in [1, 2, 3]:
            ax.axvline(
                n,
                linestyle="--",
                linewidth=0.7,
                color="black",
                alpha=0.35,
            )

        ax.set_title(
            title,
            pad=3,
        )

        ax.set_xlabel(r"$\omega/\Omega$")

        ax.set_xlim(
            0,
            5,
        )

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
        rf"$\Gamma={Gamma_fft:.2e}$",
        transform=axs[0].transAxes,
        fontsize=7,
        va="top",
    )

    fig.tight_layout(
        pad=0.4,
        w_pad=0.5,
    )

    save_figure(fig, "S_fft_spectra")


##############################################################################
# FINISHED
##############################################################################

print(f"\nSaved figures to:\n{figures_dir}\n")

plt.show()
