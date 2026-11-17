"""Inverse MDS (Kriegeskorte & Mur 2012) as implemented in rsatoolbox
``rdm.combine.rescale(method='evidence')`` followed by a weighted mean.

Re-implemented in numpy so that it can be applied to arbitrary subsets of
trials (e.g. split-half, held-out trials) without constructing RDMs objects.

Conventions
-----------
A set of trials is represented by a dense matrix ``D`` of shape
(n_trials, n_pairs) with NaN for pairs not shown in a trial. Pairs are the
upper triangle (i<j) of the full face set in a fixed order.
"""
from __future__ import annotations
import numpy as np


def pair_index(n_items: int) -> np.ndarray:
    """(n_pairs, 2) array of (i, j), i<j, in numpy triu order."""
    iu = np.triu_indices(n_items, k=1)
    return np.stack(iu, axis=1)


def pair_lookup(n_items: int) -> np.ndarray:
    """(n_items, n_items) matrix mapping (i, j) -> pair id (symmetric, diag -1)."""
    iu = np.triu_indices(n_items, k=1)
    lut = -np.ones((n_items, n_items), dtype=np.int64)
    lut[iu] = np.arange(len(iu[0]))
    lut[iu[1], iu[0]] = np.arange(len(iu[0]))
    return lut


def trial_distances(xy: np.ndarray) -> np.ndarray:
    """Upper-triangle Euclidean distances among rows of xy (n, 2)."""
    diff = xy[:, None, :] - xy[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    return d[np.triu_indices(len(xy), k=1)]


def _ss(v):
    return np.nansum(v ** 2, axis=-1, keepdims=True)


def _scale(v):
    return v / np.sqrt(_ss(v))


def _mean(v, w=None):
    if w is None:
        w = np.ones_like(v); w[np.isnan(v)] = np.nan
    return np.nansum(v * w, axis=0) / np.nansum(w, axis=0)


def inverse_mds(D: np.ndarray, method: str = "evidence", threshold: float = 1e-8,
                max_iter: int = 10000, rms_to_one: bool = True):
    """Combine partial trial RDMs into one RDM.

    Parameters
    ----------
    D : (n_trials, n_pairs) with NaN for unobserved pairs.

    Returns
    -------
    rdm : (n_pairs,) with NaN for pairs never observed.
    weights : (n_trials, n_pairs) rescaling weights (NaN where unobserved).
    n_iter : int
    """
    D_full = np.asarray(D, dtype=float)
    obs = np.isfinite(D_full).any(axis=0)
    D = D_full[:, obs]
    n_trials, n_pairs = D.shape
    if method == "evidence":
        w = (D ** 2).clip(0.2 ** 2)
    elif method == "setsize":
        setsize = np.isfinite(D).sum(axis=1)
        w = np.tile(1 / setsize, [n_pairs, 1]).T
    else:
        w = np.ones_like(D)
    w[np.isnan(D)] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        cur = _scale(_mean(D))
        prev = np.full(n_pairs, -np.inf)
        aligned = D.copy()
        it = 0
        while np.nansum((cur - prev) ** 2) > threshold and it < max_iter:
            prev = cur.copy()
            tiled = np.tile(cur, [n_trials, 1])
            tiled[np.isnan(D)] = np.nan
            aligned = _scale(D) * np.sqrt(_ss(tiled))
            cur = _scale(_mean(aligned, w))
            it += 1
        rdm_c = _mean(aligned, w)
        if rms_to_one:
            rdm_c = rdm_c / np.sqrt(np.nansum(rdm_c ** 2))
    rdm = np.full(D_full.shape[1], np.nan); rdm[obs] = rdm_c
    w_full = np.full(D_full.shape, np.nan); w_full[:, obs] = w
    return rdm, w_full, it
