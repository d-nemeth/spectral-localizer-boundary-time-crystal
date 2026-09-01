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
)
from spectral_localizer.fast_localizer import (
    FastLocalizerConfig,
    compute_idx_curve_for_gamma,
)
from spectral_localizer.mode_table import (
    print_mode_table,
)

# Parameters
N_SPINS = 10
OMEGA = 1.0
KAPPA = 1.0
MODE_IDX = 18

N_JOBS = 8


def compute_gamma_scan(
    gamma: float,
    build_L,
    K_rank_mat,
    mode_idx: int,
    x_min: float,
    x_max: float,
    cfg,
):

    print("\n" + "=" * 60)

    print(rf"Gamma = {gamma:.2f}")

    print("=" * 60)

    L_mat = build_L(gamma)

    evals, _ = np.linalg.eig(L_mat)

    all_evals = evals

    perm = print_mode_table(
        evals,
        max_modes=20,
    )

    raw_idx = perm[MODE_IDX]

    lam0 = complex(evals[raw_idx])

    print("\nSelected mode:\n")

    print(
        f"mode {MODE_IDX:2d} : "
        f"Re(lambda) = "
        f"{lam0.real:+.6f}, "
        f"Im(lambda) = "
        f"{lam0.imag:+.6f}"
    )

    gamma_out, x, idx = compute_idx_curve_for_gamma(
        gamma,
        build_L=build_L,
        X=K_rank_mat,
        lam0=lam0,
        x_min=x_min,
        x_max=x_max,
        cfg=cfg,
    )

    return {
        "gamma": gamma_out,
        "x": x,
        "idx": idx,
        "lam0": lam0,
        "evals": all_evals,
    }


def main():

    t0 = time.perf_counter()

    base_dir = Path("main_figures_datasets") / "topological_domains"

    run_dir = get_next_run_dir(base_dir)

    run_name = run_dir.name

    figures_dir = run_dir / "figures"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\nCreating run: {run_name}\n")

    params = BTCParams(
        N_spins=N_SPINS,
        omega=OMEGA,
    )

    kappa = KAPPA

    mode_idx = MODE_IDX

    gamma_list = [
        0.0,
        1.0,
        2.0,
    ]

    x_min = 0.0
    x_max = 5.0

    cfg = FastLocalizerConfig(
        kappa=kappa,
        zero_tol=1e-10,
        n_coarse=100,
        max_refine=80,
        refine_only_changes=True,
        verbose=False,
    )

    build_L = build_liouvillian_builder(params)

    _, K_rank_mat, _ = build_operator_space_coordinates(params)

    results = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        prefer="processes",
    )(
        delayed(compute_gamma_scan)(
            gamma,
            build_L,
            K_rank_mat,
            mode_idx,
            x_min,
            x_max,
            cfg,
        )
        for gamma in gamma_list
    )

    results = sorted(
        results,
        key=lambda r: r["gamma"],
    )

    x_arrays = [result["x"] for result in results]

    idx_arrays = [result["idx"] for result in results]

    selected_evals = [result["lam0"] for result in results]

    all_evals = [result["evals"] for result in results]

    data_file = run_dir / "data.npz"

    np.savez_compressed(
        data_file,
        gammas=np.array(
            gamma_list,
            dtype=float,
        ),
        x_arrays=np.array(
            x_arrays,
            dtype=object,
        ),
        idx_arrays=np.array(
            idx_arrays,
            dtype=object,
        ),
        selected_evals=np.array(
            selected_evals,
            dtype=complex,
        ),
        all_evals=np.array(
            all_evals,
            dtype=object,
        ),
        mode_idx=mode_idx,
        kappa=kappa,
    )

    print(f"\nSaved dataset:\n{data_file}")

    elapsed = time.perf_counter() - t0

    metadata = {
        "run_name": run_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": float(elapsed),
        "parallelized_over": "gamma",
        "n_jobs": N_JOBS,
        "params": asdict(params),
        "gamma_list": gamma_list,
        "mode_idx": mode_idx,
        "kappa": kappa,
        "x_min": x_min,
        "x_max": x_max,
        "fast_localizer_config": {
            "kappa": cfg.kappa,
            "zero_tol": cfg.zero_tol,
            "n_coarse": cfg.n_coarse,
            "max_refine": cfg.max_refine,
            "refine_only_changes": cfg.refine_only_changes,
        },
        "description": ("Fast spectral-localizer topological-domain scan."),
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
