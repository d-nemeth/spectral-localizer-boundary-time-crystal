from __future__ import annotations

import os

# Prevent thread oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import qutip as qt
from joblib import (
    Parallel,
    delayed,
)

from run_utils.run_manager import (
    get_next_run_dir,
)

# Parameters
Omega = 1.0

N_list = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]

Gamma_list = np.logspace(
    -2,
    -0.7,
    8,
)

# Time evolution parameters
T = 100
n_steps = 2000

times = np.linspace(
    0,
    T,
    n_steps,
)

save_raw_signals = True

# Output directory
base_dir = Path("supplemental_figures_datasets") / "spin_correlations_scaling_data"

run_dir = get_next_run_dir(base_dir)

run_name = run_dir.name

print(f"\nCreating run: {run_name}\n")

# Precompute operators for each N to avoid redundant calculations across Gamma values.
operator_cache = {}

for N in N_list:
    j = N / 2

    Jx = qt.jmat(
        j,
        "x",
    )

    Jz = qt.jmat(
        j,
        "z",
    )

    Jm = qt.jmat(
        j,
        "-",
    )

    ident = qt.qeye(N + 1)

    Jz2 = Jz @ Jz

    Jz3 = Jz2 @ Jz

    J2_val = j * (j + 1)

    H = Omega * Jx

    # Observables
    O1 = (2 / N) * Jz

    O2 = (4 / N**2) * (3 * Jz2 - J2_val * ident)

    O3 = (8 / N**3) * (5 * Jz3 - (3 * J2_val) * Jz)

    # Maximally mixed initial state to isolate sourced contributions to the dynamics
    rho0 = ident / (N + 1)

    operator_cache[N] = {
        "H": H,
        "Jm": Jm,
        "rho0": rho0,
        "e_ops": [
            O1,
            O2,
            O3,
        ],
    }


# Fourier peak extraction
def extract_peak(
    times,
    signal,
    target_omega,
    relative_window=0.20,
):
    """
    Extract harmonic peak data near target_omega using:

    - DC subtraction
    - raw real FFT (rFFT)
    - local peak search around target frequency

    Returns:
    --------
    dict with:
        peak_amplitude
        peak_frequency
        peak_omega
        peak_omega_over_Omega
    """

    signal = np.real(signal)

    # Remove DC component to prevent it from dominating the FFT
    #  and obscuring nearby peaks.
    signal = signal - np.mean(signal)

    dt = times[1] - times[0]

    fft_complex = np.fft.rfft(signal)

    freqs = np.fft.rfftfreq(
        len(signal),
        d=dt,
    )

    # Single-sided normalization
    fft_vals = 2 * np.abs(fft_complex) / len(signal)

    # Target frequency
    target_freq = target_omega / (2 * np.pi)

    # Local search window
    search_width = relative_window * target_freq

    mask = np.abs(freqs - target_freq) <= search_width

    if not np.any(mask):
        raise RuntimeError("No FFT frequencies found inside search window.")

    # Local data
    local_freqs = freqs[mask]

    local_fft = fft_vals[mask]

    # Local maximum
    local_idx = np.argmax(local_fft)

    peak_amp = local_fft[local_idx]

    peak_freq = local_freqs[local_idx]

    peak_omega = 2 * np.pi * peak_freq

    # Return peak data
    return {
        "peak_amplitude": peak_amp,
        "peak_frequency": peak_freq,
        "peak_omega": peak_omega,
        "peak_omega_over_Omega": peak_omega / Omega,
    }


def run_single_case(
    N,
    Gamma,
    solver_method="adams",
):

    print(
        f"Running N={N}, Gamma={Gamma:.5e}",
        flush=True,
    )

    cached = operator_cache[N]

    H = cached["H"]

    Jm = cached["Jm"]

    rho0 = cached["rho0"]

    e_ops = cached["e_ops"]

    c_ops = [np.sqrt(Gamma / N) * Jm]

    try:
        options = qt.Options(
            method=solver_method,
            store_states=False,
            progress_bar=False,
            nsteps=10000,
            atol=1e-9,
            rtol=1e-7,
        )

    except Exception:
        options = {
            "method": solver_method,
            "store_states": False,
            "progress_bar": False,
            "nsteps": 10000,
            "atol": 1e-9,
            "rtol": 1e-7,
        }

    # Time evolution
    result = qt.mesolve(
        H,
        rho0,
        times,
        c_ops,
        e_ops=e_ops,
        options=options,
    )

    # Signals
    sig1 = np.real(result.expect[0])

    sig2 = np.real(result.expect[1])

    sig3 = np.real(result.expect[2])

    # Extract peak data
    peak1 = extract_peak(
        times,
        sig1,
        Omega,
    )

    peak2 = extract_peak(
        times,
        sig2,
        2 * Omega,
    )

    peak3 = extract_peak(
        times,
        sig3,
        3 * Omega,
    )

    output = {
        "N": N,
        "Gamma": Gamma,
        # Amplitudes
        "A1": peak1["peak_amplitude"],
        "A2": peak2["peak_amplitude"],
        "A3": peak3["peak_amplitude"],
        # Peak omegas
        "omega1_peak": peak1["peak_omega"],
        "omega2_peak": peak2["peak_omega"],
        "omega3_peak": peak3["peak_omega"],
        # Peak omegas over Omega
        # DIMENSIONLESS PEAKS
        "omega1_over_Omega": peak1["peak_omega_over_Omega"],
        "omega2_over_Omega": peak2["peak_omega_over_Omega"],
        "omega3_over_Omega": peak3["peak_omega_over_Omega"],
    }

    if save_raw_signals:
        output["signals"] = {
            "sig1": sig1,
            "sig2": sig2,
            "sig3": sig3,
        }

    return output


# Create list of tasks for parallel execution.
tasks = [(N, Gamma) for N in N_list for Gamma in Gamma_list]

# Parellization
n_jobs = min(
    4,
    len(tasks),
)

raw_results = Parallel(
    n_jobs=n_jobs,
    backend="loky",
)(
    delayed(run_single_case)(
        N,
        Gamma,
        solver_method="adams",
    )
    for N, Gamma in tasks
)


results = {
    "data": {},
}


# Initialization of results structure
for N in N_list:
    results["data"][N] = {
        "Gamma": [],
        "A1": [],
        "A2": [],
        "A3": [],
        "omega1_peak": [],
        "omega2_peak": [],
        "omega3_peak": [],
        "omega1_over_Omega": [],
        "omega2_over_Omega": [],
        "omega3_over_Omega": [],
    }

    if save_raw_signals:
        results["data"][N]["signals"] = {
            "sig1": [],
            "sig2": [],
            "sig3": [],
        }

# Store raw results in structured format.
for item in raw_results:
    N = item["N"]

    for key in [
        "Gamma",
        "A1",
        "A2",
        "A3",
        "omega1_peak",
        "omega2_peak",
        "omega3_peak",
        "omega1_over_Omega",
        "omega2_over_Omega",
        "omega3_over_Omega",
    ]:
        results["data"][N][key].append(item[key])

    if save_raw_signals:
        results["data"][N]["signals"]["sig1"].append(item["signals"]["sig1"])

        results["data"][N]["signals"]["sig2"].append(item["signals"]["sig2"])

        results["data"][N]["signals"]["sig3"].append(item["signals"]["sig3"])


# Sort results by Gamma for each N to ensure consistent ordering.
for N in N_list:
    idx = np.argsort(results["data"][N]["Gamma"])

    for key in [
        "Gamma",
        "A1",
        "A2",
        "A3",
        "omega1_peak",
        "omega2_peak",
        "omega3_peak",
        "omega1_over_Omega",
        "omega2_over_Omega",
        "omega3_over_Omega",
    ]:
        arr = np.array(results["data"][N][key])

        results["data"][N][key] = arr[idx]

    # Sort raw signals if they are being saved
    if save_raw_signals:
        for sig_key in [
            "sig1",
            "sig2",
            "sig3",
        ]:
            sig_arr = np.array(results["data"][N]["signals"][sig_key])

            results["data"][N]["signals"][sig_key] = sig_arr[idx]

# Save results
data_file = run_dir / "data.pkl"

with open(
    data_file,
    "wb",
) as f:
    pickle.dump(
        results,
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

print(f"\nSaved dataset:\n{data_file}")

# Metadata
metadata = {
    "run_name": run_name,
    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Omega": Omega,
    "N_list": N_list,
    "Gamma_list": Gamma_list.tolist(),
    "T": T,
    "n_steps": n_steps,
    "save_raw_signals": save_raw_signals,
    "parallelization": {
        "backend": "loky",
        "n_jobs": n_jobs,
    },
    "solver": {
        "type": "qutip.mesolve",
        "method": "adams",
        "atol": 1e-9,
        "rtol": 1e-7,
        "nsteps": 10000,
    },
    "fft_method": ("real FFT (rFFT) with DC subtraction and local peak extraction"),
    "fft_peak_search_window": 0.20,
    "description": ("Spin-correlation scaling dataset for BTC harmonic response."),
}

info_file = run_dir / "info.json"

with open(
    info_file,
    "w",
) as f:
    json.dump(
        metadata,
        f,
        indent=4,
    )

print(f"Saved metadata:\n{info_file}")

print(f"\nFinished {run_name}\n")
