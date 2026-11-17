"""Participant-level trial data for Face-ID training and evaluation."""
from __future__ import annotations
import os
import numpy as np, pandas as pd
import torch
from .paths import PROC
from .invmds import trial_distances, pair_lookup, inverse_mds


class Trial:
    __slots__ = ("faces", "xy", "d", "task")
    def __init__(self, faces, xy, task="MA"):
        self.faces = np.asarray(faces, dtype=np.int64)
        self.xy = np.asarray(xy, dtype=float)
        d = trial_distances(self.xy)
        self.d = d / np.sqrt((d ** 2).mean() + 1e-12)   # RMS = 1 within trial
        self.task = task


class Participant:
    def __init__(self, pid, ma_trials, tr_trials, meta=None):
        self.pid = pid
        self.ma = ma_trials          # list[Trial]
        self.tr = tr_trials          # list[Trial] with task names TR_set{k}_rep{r}
        self.meta = meta or {}
        self.faces = np.unique(np.concatenate([t.faces for t in ma_trials])) if ma_trials else np.array([], dtype=int)


def load_experiment(exp: int, only_keep=True):
    pl = pd.read_parquet(os.path.join(PROC, f"exp{exp}_placements.parquet"))
    part = pd.read_csv(os.path.join(PROC, f"exp{exp}_participants.csv"))
    if only_keep:
        part = part[part.keep]
    n_faces = len(pd.read_csv(os.path.join(PROC, f"exp{exp}_faces.csv")))
    pl = pl[pl.pid.isin(set(part.pid))]
    out = []
    meta_rows = part.set_index("pid").to_dict("index")
    for pid, g in pl.groupby("pid", sort=True):
        ma, tr = [], []
        for (task, trial), tg in g.groupby(["task", "trial"], sort=True):
            if len(tg) < 2:
                continue
            t = Trial(tg.face_idx.to_numpy(), tg[["x", "y"]].to_numpy(), task)
            if task == "MA":
                if not tg.excluded.iloc[0]:
                    ma.append(t)
            else:
                tr.append(t)
        if len(ma) >= 2:
            out.append(Participant(pid, ma, tr, meta_rows.get(pid, {})))
    return out, n_faces


def split_trials(p: Participant, frac_est: float, seed: int, n_est: int | None = None):
    """Fixed split of a participant's MA trials into estimation / held-out."""
    rng = np.random.default_rng(hash((p.pid, seed)) % (2 ** 32))
    idx = rng.permutation(len(p.ma))
    k = int(round((frac_est if frac_est is not None else 0.7) * len(p.ma)))
    k = max(1, min(k, len(p.ma) - 1))
    est, hold = [p.ma[i] for i in idx[:k]], [p.ma[i] for i in idx[k:]]
    if n_est is not None:          # learning curve: subset of the estimation trials, SAME held-out set
        est = est[: max(1, min(n_est, len(est)))]
    return est, hold


def tensorize(trial_sets, device="cpu"):
    """trial_sets: list (one per participant slot) of lists of Trial.
    Returns dict of tensors: pi, pj, d, k, tidx, pidx; n_trials, n_part."""
    pi, pj, d, k, tidx, pidx = [], [], [], [], [], []
    t_id = 0
    for s, trials in enumerate(trial_sets):
        for t in trials:
            ii, jj = np.triu_indices(len(t.faces), k=1)
            pi.append(t.faces[ii]); pj.append(t.faces[jj]); d.append(t.d)
            k.append(np.full(len(ii), len(t.faces), dtype=float))
            tidx.append(np.full(len(ii), t_id, dtype=np.int64)); pidx.append(np.full(len(ii), s, dtype=np.int64))
            t_id += 1
    cat = lambda L, dt: torch.as_tensor(np.concatenate(L), dtype=dt, device=device) if L else torch.zeros(0, dtype=dt, device=device)
    return {"pi": cat(pi, torch.long), "pj": cat(pj, torch.long), "d": cat(d, torch.float32),
            "k": cat(k, torch.float32), "tidx": cat(tidx, torch.long), "pidx": cat(pidx, torch.long),
            "n_trials": t_id, "n_part": len(trial_sets)}


def trials_to_rdm(trials, n_faces):
    """Inverse-MDS RDM (pairs vector with NaN) from a list of Trials."""
    lut = pair_lookup(n_faces); P = n_faces * (n_faces - 1) // 2
    rows = []
    for t in trials:
        ii, jj = np.triu_indices(len(t.faces), k=1)
        row = np.full(P, np.nan); row[lut[t.faces[ii], t.faces[jj]]] = t.d; rows.append(row)
    if not rows:
        return np.full(P, np.nan)
    return inverse_mds(np.array(rows))[0]


def make_folds(participants, n_folds=10, seed=0, strat_keys=("Gender", "Race")):
    """Stratified participant folds; returns array of fold ids aligned with participants."""
    from sklearn.model_selection import StratifiedKFold
    keys = []
    for p in participants:
        keys.append("|".join(str(p.meta.get(k, "NA")) for k in strat_keys))
    keys = np.array(keys)
    # merge rare strata
    vc = pd.Series(keys).value_counts()
    keys = np.where(pd.Series(keys).map(vc).to_numpy() < n_folds, "rare", keys)
    folds = np.zeros(len(participants), dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(skf.split(np.zeros(len(keys)), keys)):
        folds[te] = f
    return folds
