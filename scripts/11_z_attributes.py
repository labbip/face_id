"""H5 / nuisance: predict perceiver attributes from z.
z coordinates differ across fold models, so for each fold model we encode ALL participants (estimation
trials, 70% split) and run a 10-fold CV classifier within that coordinate system; results are averaged over
the fold models. Labels are never used in training the models, so including training participants is valid
for attribute decoding (their z is in-sample w.r.t. the arrangement model, which is noted).
Also reports decoding from the D1 decoded weights w(z) and the in-group analysis for Exp2 (H5):
per participant, the decoded dissimilarity for same-race vs other-race face pairs.
usage: uv run python scripts/11_z_attributes.py --exp 2 --kind D1 --M 4 --tag _cv [--fold_models 0,1,2]
"""
import argparse, glob, os, sys, json, pickle
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import RESULTS, PROC
from face_id.models import FaceID
from face_id.data import split_trials, make_folds
from face_id.train import infer_z, predict_full_rdm
ap = argparse.ArgumentParser(); ap.add_argument("--exp", type=int, default=2); ap.add_argument("--kind", default="D1"); ap.add_argument("--M", type=int, default=4)
ap.add_argument("--tag", default="_cv"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--D", type=int, default=8); ap.add_argument("--fold_models", default="0,1,2,3,4")
ap.add_argument("--n_perm", type=int, default=30)
args = ap.parse_args()
parts, nf = pickle.load(open(os.path.join(PROC, f"exp{args.exp}_participants.pkl"), "rb"))
part = pd.read_csv(os.path.join(PROC, f"exp{args.exp}_participants.csv"), low_memory=False).set_index("pid")
rel = pd.read_csv(os.path.join(RESULTS, f"repro_exp{args.exp}_reliability.csv")).set_index("pid")
faces = pd.read_csv(os.path.join(PROC, f"exp{args.exp}_faces.csv"))
meta = pd.DataFrame({"pid": [p.pid for p in parts]}).join(part, on="pid").join(rel[["split_half_rho"]], on="pid")
folds = make_folds(parts, 10, 0, ("Gender", "Race") if args.exp == 1 else ("source", "version_race"))
est = [split_trials(p, 0.7, 0)[0] for p in parts]
rng = np.random.default_rng(0)
def clf(X, y, name, out):
    ok = pd.Series(y).notna().to_numpy(); yy = pd.Series(y)[ok].astype(str).to_numpy(); XX = X[ok]
    vc = pd.Series(yy).value_counts(); keep = vc[vc >= 20].index; mask = np.isin(yy, keep); yy = yy[mask]; XX = XX[mask]
    if len(set(yy)) < 2: return
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")); cv = StratifiedKFold(10, shuffle=True, random_state=0)
    acc = balanced_accuracy_score(yy, cross_val_predict(pipe, XX, yy, cv=cv))
    null = [balanced_accuracy_score(yy, cross_val_predict(pipe, XX, rng.permutation(yy), cv=cv)) for _ in range(args.n_perm)]
    out.append({"target": name, "acc": acc, "null_mean": np.mean(null), "null_sd": np.std(null), "p": np.mean(np.array(null) >= acc), "n": len(yy), "k": len(set(yy))})
def reg(X, y, name, out):
    y = pd.to_numeric(y, errors="coerce").to_numpy(); ok = np.isfinite(y)
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
    yhat = cross_val_predict(pipe, X[ok], y[ok], cv=KFold(10, shuffle=True, random_state=0))
    out.append({"target": name, "r2": 1 - ((y[ok] - yhat) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum(), "n": int(ok.sum())})
res, ingroup = [], []
for fm in [int(x) for x in args.fold_models.split(",")]:
    run = os.path.join(RESULTS, "runs", f"exp{args.exp}_{args.kind}_D{args.D}_M{args.M}_f{fm}_s{args.seed}{args.tag}")
    a = json.load(open(os.path.join(run, "args.json")))
    m = FaceID(nf, a["D"], a["M"], a["kind"], a["Dp"]).cuda(); m.load_state_dict(torch.load(os.path.join(run, "model.pt"))); m.eval()
    Z = []
    for b in range(0, len(parts), 256):
        mu, _ = infer_z(m, est[b:b + 256], "cuda"); Z.append(mu.cpu().numpy())
    Z = np.concatenate(Z)
    W = m.dec.weights(torch.as_tensor(Z).cuda()); W = W.detach().cpu().numpy().reshape(len(Z), -1) if W is not None else Z
    # residualise z on split-half reliability (nuisance) and build a reliability-only control
    rho = pd.to_numeric(meta.split_half_rho, errors="coerce").fillna(meta.split_half_rho.mean()).to_numpy()[:, None]
    A = np.hstack([np.ones_like(rho), rho]); Zres = Z - A @ np.linalg.lstsq(A, Z, rcond=None)[0]
    for feat, X in [("z", Z), ("w(z)", W), ("z|reliability", Zres), ("reliability only", rho)]:
        o = []
        clf(X, meta.gender_norm.where(meta.gender_norm.isin(["woman", "man"])), "gender", o)
        clf(X, meta.race_norm.where(meta.race_norm.isin(["white", "black", "asian"])), "race", o)
        if "source" in meta: clf(X, meta.source, "source", o)
        if "native_english" in meta: clf(X, meta.native_english.astype(str), "native_english", o)
        reg(X, meta.age_num, "age", o); reg(X, meta.split_half_rho, "split_half_rho", o)
        for r in o: r.update({"fold_model": fm, "features": feat})
        res += o
    # H5 in-group: decoded dissimilarity of same-race vs cross-race face pairs, per participant race (Exp2)
    if args.exp == 2 and "Race" in faces:
        P = predict_full_rdm(m, torch.as_tensor(Z).cuda(), nf, "cuda"); iu = np.triu_indices(nf, k=1)
        fr = faces.Race.astype(str).str.lower().to_numpy(); same = fr[iu[0]] == fr[iu[1]]
        Pz = (P - P.mean(1, keepdims=True)) / P.std(1, keepdims=True)
        for i, p in enumerate(parts):
            pr = meta.race_norm.iloc[i]
            if pr not in ("white", "black", "asian"): continue
            rec = {"pid": p.pid, "fold_model": fm, "perceiver_race": pr}
            for r_f in ("white", "black", "asian"):
                m_f = (fr[iu[0]] == r_f) & (fr[iu[1]] == r_f)  # within-race pairs of race r_f
                rec[f"within_{r_f}"] = Pz[i, m_f].mean()
            rec["within_same"] = Pz[i, same].mean(); rec["cross"] = Pz[i, ~same].mean()
            ingroup.append(rec)
res = pd.DataFrame(res)
print(res.groupby(["features", "target"])[[c for c in ["acc", "null_mean", "p", "r2"] if c in res]].mean().round(3))
res.to_csv(os.path.join(RESULTS, f"h5_attributes_exp{args.exp}_{args.kind}_M{args.M}{args.tag}.csv"), index=False)
if ingroup:
    ig = pd.DataFrame(ingroup); ig.to_csv(os.path.join(RESULTS, f"h5_ingroup_exp{args.exp}_{args.kind}_M{args.M}{args.tag}.csv"), index=False)
    g = ig.groupby(["perceiver_race"])[["within_white", "within_black", "within_asian", "within_same", "cross"]].mean()
    print("decoded (z-scored) dissimilarity among faces of each race, by perceiver race (fold models averaged):"); print(g.round(3))
    import statsmodels.formula.api as smf
    ig["own_minus_other"] = ig.within_same - ig[["within_white", "within_black", "within_asian"]].mean(axis=1)
    pm = ig.groupby("pid").own_minus_other.mean()
    from scipy.stats import ttest_1samp; print("own-race within-race dissimilarity minus mean within-race (per participant): mean %.4f, t-test p=%.2e (n=%d)" % (pm.mean(), ttest_1samp(pm, 0).pvalue, len(pm)))
