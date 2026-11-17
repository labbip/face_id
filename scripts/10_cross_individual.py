"""H3: cross-individual transfer. For fold-out participants k, l: predict k's held-out trials with z_l
(decoder of a trained model); regress rho_kl on ||z_k - z_l|| (participant-k fixed/random intercept).
Control: Euclidean distance between the two participants' estimation-trial RDMs over shared pairs
(mostly undefined since face sets differ; so we use distance in the population-embedding-projected
space instead = correlation between their predicted full RDMs).
usage: uv run python scripts/10_cross_individual.py <run_dir> [--n 150] [--direct]
"""
import os, sys, json, pickle, argparse
import numpy as np, pandas as pd, torch
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import PROC, RESULTS
from face_id.models import FaceID
from face_id.data import make_folds, split_trials, trials_to_rdm
from face_id.train import infer_z, fit_z_direct, predict_full_rdm
from face_id.rdm import nan_spearman
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--n", type=int, default=150); ap.add_argument("--direct", action="store_true")
args = ap.parse_args()
a = json.load(open(os.path.join(args.run, "args.json")))
parts, nf = pickle.load(open(os.path.join(PROC, f"exp{a['exp']}_participants.pkl"), "rb"))
folds = make_folds(parts, a["n_folds"], 0, ("Gender", "Race") if a["exp"] == 1 else ("source", "version_race"))
test = [p for p, f in zip(parts, folds) if f == a["fold"]][: args.n]
m = FaceID(nf, a["D"], a["M"], a["kind"], a["Dp"]).cuda(); m.load_state_dict(torch.load(os.path.join(args.run, "model.pt"))); m.eval()
est, hold = zip(*[split_trials(p, 0.7, 0) for p in test])
mu, _ = infer_z(m, list(est), "cuda")
if args.direct: mu = fit_z_direct(m, list(est), "cuda", steps=300, lr=0.05, l2=1e-3, init=mu)
Z = mu.cpu().numpy(); P = predict_full_rdm(m, mu, nf, "cuda"); H = [trials_to_rdm(h, nf) for h in hold]
P0 = predict_full_rdm(m, torch.zeros(1, a["M"]).cuda(), nf, "cuda")[0]
rows = []
for k in range(len(test)):
    base = nan_spearman(P0, H[k]); own = nan_spearman(P[k], H[k])
    for l in range(len(test)):
        if l == k: continue
        rows.append({"k": k, "l": l, "rho": nan_spearman(P[l], H[k]), "own": own, "popr": base,
                     "dz": float(np.linalg.norm(Z[k] - Z[l])), "dpred": float(1 - np.corrcoef(P[k], P[l])[0, 1])})
df = pd.DataFrame(rows)
print("own-z rho %.3f | other-z rho %.3f | population rho %.3f" % (df.own.mean(), df.rho.mean(), df.popr.mean()))
for x in ["dz", "dpred"]:
    df["x"] = (df[x] - df[x].mean()) / df[x].std()
    fit = smf.mixedlm("rho ~ x", df, groups=df["k"]).fit()
    print(f"mixed model rho ~ {x} (random intercept k): slope {fit.params['x']:.4f} (p={fit.pvalues['x']:.2e}); spearman(dz, rho) = {df[[x,'rho']].corr(method='spearman').iloc[0,1]:.3f}")
df.to_csv(os.path.join(RESULTS, f"cross_individual_{os.path.basename(args.run)}{'_direct' if args.direct else ''}.csv"), index=False)
