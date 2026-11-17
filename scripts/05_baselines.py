"""Non-neural baselines for fold-out participants (same split as 04_train.py, frac_est=0.7):
  agg     : training-fold aggregate RDM
  estrdm  : participant's own inverse-MDS RDM from estimation trials (seen pairs only; agg elsewhere)
  blend   : z-scored shrinkage blend w*est + (1-w)*agg, w chosen on training participants (inner split)
  rw5     : per-participant non-negative reweighting of the 5 non-metric-MDS dimensions of the training agg
            (weights fitted by maximising mean within-trial Pearson on estimation trials)
Metrics as in face_id.train.evaluate (rho_hold, rho_tr_all, rho_tr_unseen).
usage: uv run python scripts/05_baselines.py --exp 1 --fold 0
"""
import argparse, os, sys, pickle
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import PROC, RESULTS
from face_id.data import load_experiment, make_folds, split_trials, trials_to_rdm
from face_id.invmds import inverse_mds, pair_lookup
from face_id.rdm import nan_spearman, vec_to_mat
from face_id.train import tr_targets

ap = argparse.ArgumentParser(); ap.add_argument("--exp", type=int, default=1); ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--n_folds", type=int, default=10); ap.add_argument("--frac_est", type=float, default=0.7); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
cache = os.path.join(PROC, f"exp{args.exp}_participants.pkl")
parts, n_faces = pickle.load(open(cache, "rb")) if os.path.exists(cache) else load_experiment(args.exp)
strat = ("Gender", "Race") if args.exp == 1 else ("source", "version_race")
folds = make_folds(parts, args.n_folds, seed=0, strat_keys=strat)
train = [p for p, f in zip(parts, folds) if f != args.fold]; test = [p for p, f in zip(parts, folds) if f == args.fold]
ind = np.load(os.path.join(RESULTS, f"repro_exp{args.exp}_individual_RDMs.npy"))
pid2row = {p: i for i, p in enumerate(pd.read_csv(os.path.join(RESULTS, f"repro_exp{args.exp}_individual_RDMs_pids.csv")).pid)}
agg = inverse_mds(ind[[pid2row[p.pid] for p in train if p.pid in pid2row]])[0]
LUT = pair_lookup(n_faces)

def zs(v):
    o = np.isfinite(v); r = np.full_like(v, np.nan); r[o] = (v[o] - v[o].mean()) / (v[o].std() + 1e-9); return r
za = zs(agg)

# --- 5-D non-metric MDS of agg (sklearn SMACOF, metric=False) ---
from sklearn.manifold import MDS
Mx = vec_to_mat(np.nan_to_num(agg, nan=np.nanmean(agg)), n_faces)
X5 = MDS(n_components=5, metric=False, dissimilarity="precomputed", random_state=0, n_init=4, max_iter=1000, normalized_stress="auto").fit_transform(Mx)
X5 = X5 - X5.mean(0); X5 = X5 @ np.linalg.svd(X5, full_matrices=False)[2].T  # principal axes
np.save(os.path.join(RESULTS, f"baseline_exp{args.exp}_f{args.fold}_mds5.npy"), X5)

def rw5_fit(trials):
    def negcorr(logw):
        w = np.exp(logw); c = []
        for t in trials:
            ii, jj = np.triu_indices(len(t.faces), k=1)
            dl = np.sqrt((((X5[t.faces[ii]] - X5[t.faces[jj]]) ** 2) * w).sum(1))
            if dl.std() > 0: c.append(np.corrcoef(dl, t.d)[0, 1])
        return -np.mean(c) + 1e-3 * (logw ** 2).sum()
    r = minimize(negcorr, np.zeros(5), method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-5})
    return np.exp(r.x)

def rw5_rdm(w):
    iu = np.triu_indices(n_faces, k=1)
    return np.sqrt((((X5[iu[0]] - X5[iu[1]]) ** 2) * w).sum(1))

# --- choose blend weight on training participants (inner evaluation) ---
rng = np.random.default_rng(1)
sub = [train[i] for i in rng.choice(len(train), 200, replace=False)]
ws = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
sc = {w: [] for w in ws}
for p in sub:
    est, hold = split_trials(p, args.frac_est, args.seed)
    ze = zs(trials_to_rdm(est, n_faces)); h = trials_to_rdm(hold, n_faces)
    for w in ws:
        b = np.where(np.isfinite(ze), w * ze + (1 - w) * za, za); sc[w].append(nan_spearman(b, h))
w_best = max(ws, key=lambda w: np.nanmean(sc[w]))
print("blend weight chosen on training participants:", w_best, {w: round(np.nanmean(sc[w]), 3) for w in ws})

rows = []
for p in test:
    est, hold = split_trials(p, args.frac_est, args.seed)
    e = trials_to_rdm(est, n_faces); h = trials_to_rdm(hold, n_faces); ze = zs(e)
    preds = {"agg": agg, "estrdm": np.where(np.isfinite(e), zs(e), za), "blend": np.where(np.isfinite(ze), w_best * ze + (1 - w_best) * za, za)}
    w5 = rw5_fit(est); preds["rw5"] = rw5_rdm(w5); preds["mds5"] = rw5_rdm(np.ones(5))
    r = {"pid": p.pid, "fold": args.fold, "n_est": len(est), "n_hold": len(hold)}
    r.update({f"w5_{k}": w5[k] for k in range(5)})
    trs = tr_targets(p, n_faces) if p.tr else []
    for name, pr in preds.items():
        r[f"rho_hold_{name}"] = nan_spearman(pr, h)
        if trs:
            ra, ru = [], []
            for t in trs:
                y = t["rdm"][t["obs"]]; x = pr[t["obs"]]; ra.append(nan_spearman(x, y))
                if (~t["both_seen"]).sum() >= 5: ru.append(nan_spearman(x[~t["both_seen"]], y[~t["both_seen"]]))
            r[f"rho_tr_all_{name}"] = np.nanmean(ra) if ra else np.nan; r[f"rho_tr_unseen_{name}"] = np.nanmean(ru) if ru else np.nan
    rows.append(r)
df = pd.DataFrame(rows)
os.makedirs(os.path.join(RESULTS, "baselines"), exist_ok=True)
df.to_csv(os.path.join(RESULTS, "baselines", f"exp{args.exp}_f{args.fold}.csv"), index=False)
cols = [c for c in df.columns if c.startswith("rho_")]
print(df[cols].mean().round(3).to_string())
