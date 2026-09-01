from __future__ import annotations

import numpy as np

from spectral_localizer.mode_table import (
    sort_modes_by_real_part,
)


def compute_rank_weights(
    V: np.ndarray,
    B: np.ndarray,
    kq_labels: list[tuple[int, int]],
):
    """
    Compute tensor-rank weight distributions of eigenmodes.

    Parameters
    ----------
    V :
        Matrix whose columns are eigenvectors. This may contain
        either right or left eigenvectors of the Liouvillian.

    B :
        Matrix whose columns are the physical basis vectors
        written in the computational/operator basis.

        In the spherical tensor construction, the columns of B
        correspond to vectorized tensor operators |T_q^k>>.

        The projection

            B^\dagger V

        therefore gives the expansion coefficients of each
        eigenmode in the spherical tensor basis.

    kq_labels :
        Labels associated with each basis vector in B.

        Each entry is a tuple (k, q), where:
            - k is the tensor rank (physical sector)
            - q labels basis states within that sector

    Returns
    -------
    rank_weights :
        Array whose entry

            rank_weights[a, k]

        gives the normalized weight carried by eigenmode a
        in tensor-rank sector k.

        Explicitly,

            w_k^{(a)}
            =
            \sum_q |<<T_q^k | psi_a>>|^2.

    k_list :
        Sorted list of tensor-rank sectors.
    """

    coeffs = B.conj().T @ V
    # The entry coeffs[i, a] gives the expansion coefficient
    #  of mode a in the basis vector B[:, i].
    weights = np.abs(coeffs) ** 2

    # Create a dictionary mapping each tensor rank k to the indices of the
    # corresponding basis vectors in B.
    k_list = sorted(set(k for (k, q) in kq_labels))

    k_to_indices = {k: [] for k in k_list}

    for i, (k, q) in enumerate(kq_labels):
        k_to_indices[k].append(i)

    # Initialize an array to hold the rank weights of each mode.
    # The entry rank_weights[a, m] will give the weight
    #  of mode a in tensor-rank sector k_list[m].
    rank_weights = np.zeros(
        (
            V.shape[1],
            len(k_list),
        ),
        dtype=float,
    )

    # Sum the weights of each mode over the basis vectors corresponding
    #  to each tensor rank k.
    for ik, k in enumerate(k_list):
        # Get the indices of the basis vectors in B that correspond
        #  to tensor rank k.
        idxs = k_to_indices[k]
        # Sum over basis vectors in the k sector to get the
        #  total weight of each mode in that sector.
        rank_weights[:, ik] = weights[idxs, :].sum(axis=0)
    # Normalize the rank weights of each mode to sum to 1.
    rank_weights /= np.maximum(
        rank_weights.sum(
            axis=1,
            keepdims=True,
        ),
        1e-16,
    )

    return rank_weights, k_list


def compute_left_right_rank_profiles(
    L_mat: np.ndarray,
    B: np.ndarray,
    kq_labels: list[tuple[int, int]],
):
    """
    Compute left/right Liouvillian eigensystems together with
    their tensor-rank delocalization profiles.

    Parameters
    ----------
    L_mat :
        Matrix representation of the Liouvillian superoperator.

    B :
        Matrix whose columns are the physical basis vectors
        (e.g. vectorized spherical tensor operators).

    kq_labels :
        Labels identifying the tensor-rank structure of the
        physical basis vectors.

    Returns
    -------
    Dictionary containing:

    evals :
        Sorted Liouvillian eigenvalues.

    R :
        Sorted right eigenvectors.

    L :
        Left eigenvectors matched to the sorted right modes.

    rank_weights_R :
        Tensor-rank weight distributions of right eigenmodes.

    rank_weights_L :
        Tensor-rank weight distributions of left eigenmodes.

    k_list :
        Sorted list of tensor-rank sectors.

    perm :
        Permutation used to physically sort the right modes.
    """

    # Right eigensystem
    evals_R, R = np.linalg.eig(L_mat)

    # Left eigensystem
    evals_L, L = np.linalg.eig(L_mat.conj().T)

    evals_R = evals_R.astype(complex)
    evals_L = evals_L.astype(complex)

    R = R.astype(complex)
    L = L.astype(complex)

    # Normalize modes to unit norm.
    R /= np.maximum(
        np.linalg.norm(
            R,
            axis=0,
            keepdims=True,
        ),
        1e-16,
    )

    L /= np.maximum(
        np.linalg.norm(
            L,
            axis=0,
            keepdims=True,
        ),
        1e-16,
    )

    # Physical sorting of RIGHT modes
    perm = sort_modes_by_real_part(
        evals_R,
    )

    evals_sorted = evals_R[perm]

    R_sorted = R[:, perm]

    # Match LEFT modes to RIGHT modes
    L_sorted = np.zeros_like(R_sorted)

    # We match left and right modes by comparing their eigenvalues.
    unused = list(range(len(evals_L)))

    for a, lam in enumerate(evals_sorted):
        # Find the left mode that corresponds to the right mode a.
        distances = np.abs(evals_L[unused] - lam.conjugate())
        # Find the index of the left mode whose eigenvalue
        #  is closest to the complex conjugate of lam.
        j_local = int(np.argmin(distances))
        # Remove the matched left mode from the list of unused modes.
        j = unused.pop(j_local)
        # Assign the matched left mode to the same column index
        #  a as the right mode.
        L_sorted[:, a] = L[:, j]

    # Rank profiles
    rank_weights_R, k_list = compute_rank_weights(
        R_sorted,
        B,
        kq_labels,
    )

    rank_weights_L, _ = compute_rank_weights(
        L_sorted,
        B,
        kq_labels,
    )

    return {
        "evals": evals_sorted,
        "R": R_sorted,
        "L": L_sorted,
        "rank_weights_R": rank_weights_R,
        "rank_weights_L": rank_weights_L,
        "k_list": k_list,
        "perm": perm,
    }
