from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from run_utils.run_manager import (
    get_next_run_dir,
)
from spectral_localizer import (
    BTCParams,
    build_liouvillian_builder,
    build_operator_space_coordinates,
    localizer_gap_and_index,
    spectral_localizer,
)


def main():

    t0 = time.perf_counter()

    # MODE = "gamma_scan"
    MODE = "kappa_scan"

    base_dir = Path("supplemental_figures_datasets") / "localizer_scans"

    # Create dynamic run directory
    run_dir = get_next_run_dir(base_dir)

    RUN = run_dir.name

    figures_dir = run_dir / "figures"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\nCreating run: {RUN}\n")

    # Parameters
    params = BTCParams(
        N_spins=5,
        omega=1.0,
    )

    lam0 = 0.0 + 0.0j

    zero_tol = 1e-8

    Nk = 80 * (params.N_spins + 1)

    # Sweep configuration
    fixed_gamma = None
    fixed_kappa = None

    if MODE == "gamma_scan":
        sweep_parameter = "gamma"

        sweep_vals = np.linspace(
            0.0,
            2.0,
            5,
        )

        fixed_kappa = 1.0

    elif MODE == "kappa_scan":
        sweep_parameter = "kappa"

        sweep_vals = np.array(
            [
                0.01,
                0.1,
                0.5,
                1.0,
                2.0,
                5.0,
            ]
        )

        fixed_gamma = 1.0

    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    # Build model
    build_L = build_liouvillian_builder(params)

    _, K_rank_mat, _ = build_operator_space_coordinates(params)

    # x0 grid
    k_eigs = np.linalg.eigvalsh(K_rank_mat).real

    x0_vals = np.linspace(
        float(k_eigs.min()),
        float(k_eigs.max()),
        Nk,
    )

    # Containers
    mu_vals_all = []

    idx_vals_all = []

    # Main sweep
    for sweep_val in sweep_vals:
        if sweep_parameter == "gamma":
            gamma = float(sweep_val)

            kappa = fixed_kappa

        else:
            gamma = fixed_gamma

            kappa = float(sweep_val)

        print(f"Computing {sweep_parameter} = {sweep_val:.4f}")

        L_mat = build_L(gamma)

        mu_vals = np.zeros(
            Nk,
            dtype=float,
        )

        idx_vals = np.zeros(
            Nk,
            dtype=int,
        )

        for i, x0 in enumerate(x0_vals):
            L_loc = spectral_localizer(
                L_mat,
                K_rank_mat,
                lam0=lam0,
                x0=x0,
                kappa=kappa,
            )

            mu, idx = localizer_gap_and_index(
                L_loc,
                zero_tol=zero_tol,
            )

            mu_vals[i] = mu

            idx_vals[i] = idx

        mu_vals_all.append(mu_vals)

        idx_vals_all.append(idx_vals)

    # Convert arrays
    mu_vals_all = np.array(
        mu_vals_all,
        dtype=float,
    )

    idx_vals_all = np.array(
        idx_vals_all,
        dtype=int,
    )

    sweep_vals = np.array(
        sweep_vals,
        dtype=float,
    )

    # Save dataset
    data_file = run_dir / "data.npz"

    np.savez_compressed(
        data_file,
        x0_vals=x0_vals,
        sweep_vals=sweep_vals,
        mu_vals=mu_vals_all,
        idx_vals=idx_vals_all,
    )

    print(f"\nSaved dataset:\n{data_file}")

    # Metadata
    elapsed = time.perf_counter() - t0

    metadata = {
        "run_name": RUN,
        "run_dir": str(run_dir),
        "figures_dir": str(figures_dir),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": elapsed,
        "mode": MODE,
        "sweep_parameter": sweep_parameter,
        "sweep_values": sweep_vals.tolist(),
        "fixed_gamma": fixed_gamma,
        "fixed_kappa": fixed_kappa,
        "lambda0_real": float(lam0.real),
        "lambda0_imag": float(lam0.imag),
        "zero_tol": float(zero_tol),
        "Nk": int(Nk),
        "params": asdict(params),
        "description": ("1D spectral localizer diagnostics versus x0."),
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

    print(f"\nFinished {RUN}\nin {elapsed:.2f} s\n")


if __name__ == "__main__":
    main()
