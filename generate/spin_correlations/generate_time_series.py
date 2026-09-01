from __future__ import annotations

import os

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

N_list = [10, 320]
Gamma_list = [0.1, 0.5, 1]

# Time evolution parameters
T = 100
n_steps = 10000
times = np.linspace(
    0,
    T,
    n_steps,
)

save_raw_signals = True

# Output directory
base_dir = Path("supplemental_figures_datasets") / "spin_correlations_time_series_data"

run_dir = get_next_run_dir(base_dir)

run_name = run_dir.name

print(f"\nCreating run: {run_name}\n")

# Precompute operators
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

    # Maximally mixed initial state to isolate sourced contributions
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
):

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

    target_freq = target_omega / (2 * np.pi)

    idx = np.argmin(np.abs(freqs - target_freq))
    amp = fft_vals[idx]

    return amp


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

    # Fourier amplitudes
    A1 = extract_peak(
        times,
        sig1,
        Omega,
    )

    A2 = extract_peak(
        times,
        sig2,
        2 * Omega,
    )

    A3 = extract_peak(
        times,
        sig3,
        3 * Omega,
    )

    # Output
    output = {
        "N": N,
        "Gamma": Gamma,
        "A1": A1,
        "A2": A2,
        "A3": A3,
    }

    if save_raw_signals:
        output["signals"] = {
            "sig1": sig1,
            "sig2": sig2,
            "sig3": sig3,
        }

    return output


tasks = [(N, Gamma) for N in N_list for Gamma in Gamma_list]

# Parellization
n_jobs = min(
    6,
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


# Storage
for N in N_list:
    results["data"][N] = {
        "Gamma": [],
        "A1": [],
        "A2": [],
        "A3": [],
    }

    if save_raw_signals:
        results["data"][N]["signals"] = {
            "sig1": [],
            "sig2": [],
            "sig3": [],
        }


for item in raw_results:
    N = item["N"]

    results["data"][N]["Gamma"].append(item["Gamma"])

    results["data"][N]["A1"].append(item["A1"])

    results["data"][N]["A2"].append(item["A2"])

    results["data"][N]["A3"].append(item["A3"])

    if save_raw_signals:
        results["data"][N]["signals"]["sig1"].append(item["signals"]["sig1"])

        results["data"][N]["signals"]["sig2"].append(item["signals"]["sig2"])

        results["data"][N]["signals"]["sig3"].append(item["signals"]["sig3"])

# Sort by Gamma
for N in N_list:
    idx = np.argsort(results["data"][N]["Gamma"])

    for key in [
        "Gamma",
        "A1",
        "A2",
        "A3",
    ]:
        arr = np.array(results["data"][N][key])

        results["data"][N][key] = arr[idx]

    if save_raw_signals:
        for sig_key in [
            "sig1",
            "sig2",
            "sig3",
        ]:
            sig_arr = np.array(results["data"][N]["signals"][sig_key])

            results["data"][N]["signals"][sig_key] = sig_arr[idx]

# Save data
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
    "Gamma_list": Gamma_list,
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
    "observables": {
        "A1": "rank-1 intensive observable",
        "A2": ("rank-2 intensive irreducible tensor-like observable"),
        "A3": ("rank-3 intensive irreducible tensor-like observable"),
    },
    "frequency_channels": {
        "A1": "Omega",
        "A2": "2 Omega",
        "A3": "3 Omega",
    },
    "fft_method": ("unwindowed FFT with DC subtraction"),
    "description": (
        "BTC time-series dataset for representative dissipation strengths."
    ),
}

# Save metadata
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
