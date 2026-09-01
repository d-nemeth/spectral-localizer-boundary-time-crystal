from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def sort_modes_by_real_part(
    evals: np.ndarray,
    decimals: int = 9,
):
    """
    Return permutation that sorts eigenmodes by:

        1) decreasing real part
        2) increasing imaginary part

    Real parts are rounded before sorting in order
    to avoid floating-point noise preventing the
    secondary imaginary-part ordering.
    """

    evals = np.asarray(
        evals,
        dtype=complex,
    )

    re = np.round(
        evals.real,
        decimals=decimals,
    )

    im = evals.imag

    perm = np.lexsort(
        (
            im,  # secondary key
            -re,  # primary key
        )
    )

    return perm


def print_mode_table(
    evals: np.ndarray,
    max_modes: int | None = None,
    precision: int = 5,
):
    """
    Print Liouvillian eigenmodes sorted
    by decreasing real part and then
    increasing imaginary part.
    """

    evals = np.asarray(
        evals,
        dtype=complex,
    )

    perm = sort_modes_by_real_part(evals)

    if max_modes is not None:
        perm = perm[:max_modes]

    fmt = f"{{: .{precision}f}}"

    print("\nLiouvillian eigenmodes:")

    print(
        f"{'mode':>6} "
        f"{'Re(lambda)':>18} "
        f"{'Im(lambda)':>18} "
        f"{'|Im(lambda)|':>18} "
        f"{'|lambda|':>18}"
    )

    print("-" * 90)

    for physical_mode, raw_idx in enumerate(perm):
        lam = evals[raw_idx]

        print(
            f"{physical_mode:6d} "
            f"{fmt.format(lam.real):>18} "
            f"{fmt.format(lam.imag):>18} "
            f"{fmt.format(abs(lam.imag)):>18} "
            f"{fmt.format(abs(lam)):>18}"
        )

    return perm


def build_mode_dataframe(
    evals: np.ndarray,
    precision: int = 5,
):
    """
    Build a pandas dataframe containing the
    sorted Liouvillian eigenmode table.
    """

    evals = np.asarray(
        evals,
        dtype=complex,
    )

    perm = sort_modes_by_real_part(evals)

    evals_sorted = evals[perm]

    rows = []

    for physical_mode, lam in enumerate(evals_sorted):
        rows.append(
            {
                "mode": physical_mode,
                "Re(lambda)": np.round(
                    lam.real,
                    precision,
                ),
                "Im(lambda)": np.round(
                    lam.imag,
                    precision,
                ),
                "|Im(lambda)|": np.round(
                    abs(lam.imag),
                    precision,
                ),
                "|lambda|": np.round(
                    abs(lam),
                    precision,
                ),
            }
        )

    df = pd.DataFrame(rows)

    return df


def save_mode_table(
    evals: np.ndarray,
    out_dir: Path,
    filename: str = "eigvals",
    precision: int = 5,
):
    """
    Save Liouvillian eigenmode table in both:

        - CSV format
        - plain-text table format

    Parameters
    ----------
    evals :
        Liouvillian eigenvalues.

    out_dir :
        Output directory.

    filename :
        Base filename without extension.

    precision :
        Decimal precision used in saved tables.
    """

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = build_mode_dataframe(
        evals,
        precision=precision,
    )

    csv_file = out_dir / f"{filename}.csv"

    txt_file = out_dir / f"{filename}.txt"

    df.to_csv(
        csv_file,
        index=False,
    )

    with open(
        txt_file,
        "w",
    ) as f:
        f.write(df.to_string(index=False))

    print(f"\nSaved mode table:\n{csv_file}\n{txt_file}")

    return df
