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
)
from spectral_localizer.kq_basis import (
    build_kq_basis_from_casimir_and_Q,
)
from spectral_localizer.mode_tools import (
    compute_left_right_rank_profiles,
)


def main():

    t0 = time.perf_counter()

    # Base output directory

    base_dir = Path("main_figures_datasets/delocalization")

    # Create run directory
    out_dir = get_next_run_dir(base_dir)

    RUN = out_dir.name

    print(f"\nCreating dataset: {RUN}\n")

    # Parameters
    params = BTCParams(
        N_spins=5,
        omega=1.0,
    )

    gammas = [
        1.0,
        5.0,
    ]

    mode_idx = 8

    # Build Liouvillian
    build_L = build_liouvillian_builder(params)

    K2_mat, _, Q_mat = build_operator_space_coordinates(params)

    # Build spherical tensor basis
    kq_basis_vecs, kq_labels, _ = build_kq_basis_from_casimir_and_Q(
        K2_mat,
        Q_mat,
        params.j,
    )

    B = np.column_stack(kq_basis_vecs)

    # Containers
    evals_all = []

    rank_weights_R_all = []

    rank_weights_L_all = []

    # Generate datasets
    for gamma in gammas:
        print(f"Generating data for Gamma = {gamma:.2f}")

        # Liouvillian
        L_mat = build_L(gamma)

        # Compute profiles
        data = compute_left_right_rank_profiles(
            L_mat,
            B,
            kq_labels,
        )

        evals_sorted = data["evals"]

        rank_weights_R = data["rank_weights_R"]

        rank_weights_L = data["rank_weights_L"]

        # Store results
        evals_all.append(evals_sorted)

        rank_weights_R_all.append(rank_weights_R)

        rank_weights_L_all.append(rank_weights_L)

    # Convert to arrays
    evals_all = np.array(
        evals_all,
        dtype=complex,
    )

    rank_weights_R_all = np.array(
        rank_weights_R_all,
        dtype=float,
    )

    rank_weights_L_all = np.array(
        rank_weights_L_all,
        dtype=float,
    )

    k_list = np.array(
        data["k_list"],
        dtype=int,
    )

    gammas_array = np.array(
        gammas,
        dtype=float,
    )

    # Save dataset
    data_file = out_dir / "data.npz"

    np.savez_compressed(
        data_file,
        gammas=gammas_array,
        k_list=k_list,
        evals=evals_all,
        rank_weights_R=rank_weights_R_all,
        rank_weights_L=rank_weights_L_all,
        mode_idx=mode_idx,
    )

    print(f"Saved dataset:\n{data_file}")

    # Save metadata
    elapsed = time.perf_counter() - t0

    metadata = {
        "run_name": RUN,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": elapsed,
        "params": asdict(params),
        "gammas": gammas,
        "mode_idx": mode_idx,
        "basis": "spherical_tensor_basis",
        "description": (
            "Tensor-rank delocalization profiles for left/right Liouvillian eigenmodes."
        ),
    }

    metadata_file = out_dir / "info.json"

    with open(
        metadata_file,
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
        )

    print(f"Saved metadata:\n{metadata_file}")

    print(f"\nFinished {RUN} in {elapsed:.2f} s\n")


if __name__ == "__main__":
    main()
