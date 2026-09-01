from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

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

N_JOBS = 8


def compute_kappa_scan(
    kappa: float,
    L_mat: np.ndarray,
    X: np.ndarray,
    x0: float,
    zero_tol: float,
    re_grid: np.ndarray,
    im_grid: np.ndarray,
):

    print(f"Starting kappa = {kappa:g}")

    mu_map = np.zeros(
        (
            len(im_grid),
            len(re_grid),
        ),
        dtype=float,
    )

    idx_map = np.zeros(
        (
            len(im_grid),
            len(re_grid),
        ),
        dtype=int,
    )

    for yi, im0 in enumerate(im_grid):
        for xi, re0 in enumerate(re_grid):
            lam0 = complex(
                re0,
                im0,
            )

            L_loc = spectral_localizer(
                L_mat,
                X,
                lam0=lam0,
                x0=x0,
                kappa=float(kappa),
            )

            mu, idx = localizer_gap_and_index(
                L_loc,
                zero_tol=zero_tol,
            )

            mu_map[yi, xi] = mu
            idx_map[yi, xi] = idx

    print(f"Finished kappa = {kappa:g} | unique nu = {np.unique(idx_map)}")

    return mu_map, idx_map


def main():

    t0 = time.perf_counter()

    base_dir = Path("supplemental_figures_datasets") / "localizer_islands"

    run_dir = get_next_run_dir(base_dir)
    run_name = run_dir.name

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\nCreating run: {run_name}\n")

    params = BTCParams(
        N_spins=10,
        omega=1.0,
    )

    gamma = 1.0

    kappa_vals = np.array(
        [
            0.01,
            0.1,
            0.5,
            1.0,
            2.0,
            5.0,
        ]
    )

    x0 = 1.0
    zero_tol = 1e-8

    re_grid = np.linspace(
        -0.15,
        0.0,
        200,
    )

    im_grid = np.linspace(
        -1.5,
        1.5,
        200,
    )

    build_L = build_liouvillian_builder(params)

    _, K_rank_mat, _ = build_operator_space_coordinates(params)

    X = K_rank_mat

    L_mat = build_L(gamma)

    eigvals = np.linalg.eigvals(L_mat)

    results = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        prefer="processes",
    )(
        delayed(compute_kappa_scan)(
            float(kappa),
            L_mat,
            X,
            x0,
            zero_tol,
            re_grid,
            im_grid,
        )
        for kappa in kappa_vals
    )

    mu_all = np.array(
        [result[0] for result in results],
        dtype=float,
    )

    idx_all = np.array(
        [result[1] for result in results],
        dtype=int,
    )

    data_file = run_dir / "data.npz"

    np.savez_compressed(
        data_file,
        re_grid=re_grid,
        im_grid=im_grid,
        kappa_vals=kappa_vals,
        mu_vals=mu_all,
        idx_vals=idx_all,
        eigvals=eigvals,
    )

    print(f"\nSaved dataset:\n{data_file}")

    elapsed = time.perf_counter() - t0

    metadata = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "figures_dir": str(figures_dir),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": float(elapsed),
        "diagnostic": "localizer_islands_scan",
        "parallelized_over": "kappa",
        "n_jobs": int(N_JOBS),
        "params": asdict(params),
        "gamma": float(gamma),
        "kappa_vals": kappa_vals.tolist(),
        "x0": float(x0),
        "zero_tol": float(zero_tol),
        "re_grid_min": float(re_grid.min()),
        "re_grid_max": float(re_grid.max()),
        "re_grid_points": int(len(re_grid)),
        "im_grid_min": float(im_grid.min()),
        "im_grid_max": float(im_grid.max()),
        "im_grid_points": int(len(im_grid)),
        "description": (
            "Spectral-localizer island scan in the complex "
            "lambda0 plane for different kappa values."
        ),
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

    print(f"\nFinished {run_name} in {elapsed:.2f} s\n")


if __name__ == "__main__":
    main()
