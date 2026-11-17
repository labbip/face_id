"""RDM helpers."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr


def vec_to_mat(v: np.ndarray, n: int, diag=0.0) -> np.ndarray:
    m = np.full((n, n), diag, dtype=float)
    iu = np.triu_indices(n, k=1)
    m[iu] = v
    m[iu[1], iu[0]] = v
    return m


def mat_to_vec(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices(m.shape[0], k=1)]


def nan_spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return spearmanr(a[ok], b[ok]).statistic


def rms_to_one(v: np.ndarray) -> np.ndarray:
    return v / np.sqrt(np.nansum(v ** 2))
