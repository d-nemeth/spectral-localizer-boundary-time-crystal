from __future__ import annotations

from pathlib import Path

import numpy as np

from spectral_localizer.btc_model import (
    BTCParams,
    build_liouvillian_builder,
)
from spectral_localizer.mode_table import (
    print_mode_table,
    save_mode_table,
)


def main():

    params = BTCParams(
        N_spins=10,
        omega=1.0,
    )

    gamma = 5.0

    # Build Liouvillian
    build_liouvillian = build_liouvillian_builder(params)

    L_mat = build_liouvillian(gamma)

    # Eigensystem
    evals, _ = np.linalg.eig(L_mat)

    # Print table to terminal
    perm = print_mode_table(
        evals,
        max_modes=40,
        precision=5,
    )

    # Save mode tables to file
    out_dir = Path("main_figures_datasets") / "eigenmode_tables"

    save_mode_table(
        evals,
        out_dir=out_dir,
        filename=(f"N_{params.N_spins}_gamma_{gamma:.2f}"),
        precision=6,
    )


if __name__ == "__main__":
    main()
