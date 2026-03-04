from __future__ import annotations
import numpy as np

def sort_modes_steady_then_absRe(evals: np.ndarray, R: np.ndarray):
    steady = int(np.argmin(np.abs(evals)))
    idx = np.arange(len(evals))
    rest = idx[idx != steady]
    rest_sorted = rest[np.lexsort((
        np.abs(evals[rest]),
        np.abs(evals[rest].imag),
        np.abs(evals[rest].real),
    ))]
    perm = np.concatenate(([steady], rest_sorted))
    return evals[perm], R[:, perm], perm

def pick_three_modes_sorted(evals_sorted: np.ndarray, Nslow=40, mid_frac=0.55, pick="large_real"):
    n = len(evals_sorted)
    steady = 0

    slow = np.arange(min(Nslow, n))
    slow_no0 = slow[slow != 0]
    slow_osc = int(slow_no0[np.argmax(np.abs(evals_sorted.imag[slow_no0]))]) if slow_no0.size else 1

    start = int(max(1, np.floor(mid_frac * n)))
    stop  = int(max(start + 1, np.floor(0.85 * n)))
    cand = np.arange(start, stop)
    cand = cand[(cand != steady) & (cand != slow_osc)]
    if cand.size == 0:
        cand = np.setdiff1d(np.arange(n), np.array([steady, slow_osc]))
        if cand.size == 0:
            return [steady, slow_osc, slow_osc]

    if pick == "large_real":
        third = int(cand[np.argmax(evals_sorted.real[cand])])
    elif pick == "large_absreal":
        third = int(cand[np.argmax(np.abs(evals_sorted.real[cand]))])
    else:
        raise ValueError("pick must be 'large_real' or 'large_absreal'")

    return [steady, slow_osc, third]

def compute_rank_weights_norm_for_gamma(L_mat: np.ndarray, B: np.ndarray, kq_labels: list[tuple[int,int]]):
    evals, R = np.linalg.eig(L_mat)
    evals = evals.astype(complex)
    R = R.astype(complex)
    R /= np.maximum(np.linalg.norm(R, axis=0, keepdims=True), 1e-16)

    evals, R, _ = sort_modes_steady_then_absRe(evals, R)

    coeffs = B.conj().T @ R
    weights = np.abs(coeffs) ** 2

    k_list = sorted(set(k for (k, q) in kq_labels))
    k_to_indices = {k: [] for k in k_list}
    for a, (k, q) in enumerate(kq_labels):
        k_to_indices[k].append(a)

    rank_weights = np.zeros((len(evals), len(k_list)), dtype=float)
    for ik, k in enumerate(k_list):
        idxs = k_to_indices[k]
        rank_weights[:, ik] = weights[idxs, :].sum(axis=0)

    rank_weights_norm = rank_weights / np.maximum(rank_weights.sum(axis=1, keepdims=True), 1e-16)
    return evals, rank_weights_norm, k_list
