"""Model-free suitability screen for candidate datasets: can a rater's idiosyncratic ratings (residual after item and rater means)
be predicted from other raters? For held-out raters (10-fold), take n_est of their observed items, fit ridge of their residuals on the
training raters' residual profiles (collaborative filtering), predict the rest; report Pearson r vs the population item mean.
Also reports the variance share of the residual. Usage: uv run --with openpyxl --with xlrd python -I scripts/64_cf_screen.py <src> [max_users]
src: london | lapis | aadb | eva | scut | jester | keller
"""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
src = sys.argv[1]; MAXU = int(sys.argv[2]) if len(sys.argv) > 2 else 300
def long_to_mat(d, min_rat=100):
    d = d.groupby(["participant_id", "image_id"], as_index=False).rating.mean(); n = d.groupby("participant_id").size(); d = d[d.participant_id.isin(n[n >= min_rat].index)]
    return d.pivot(index="participant_id", columns="image_id", values="rating").to_numpy(float)
if src == "london":
    r = pd.read_csv("data/raw/london/london_faces_ratings.csv"); R = r.iloc[:, 3:].to_numpy(float); R = R[R.std(1) > 0]
elif src == "lapis":
    R = long_to_mat(pd.read_csv("data/raw/lapis/piaa.csv", low_memory=False))
elif src == "aadb":
    R = long_to_mat(pd.read_csv("data/processed/aadb_ratings.csv"))
elif src == "eva":
    v = pd.read_csv("data/raw/eva/votes_filtered.csv", sep="=").rename(columns={"user_id": "participant_id", "score": "rating"}); R = long_to_mat(v)
elif src == "scut":
    d = pd.read_excel("data/raw/scut/unzip/SCUT-FBP5500_v2/All_Ratings.xlsx", sheet_name="ALL"); R = d.pivot(index="Rater", columns="Filename", values="Rating").to_numpy(float)
elif src == "jester":
    d = pd.read_excel("data/raw/jester/jokes_ratings.xls", header=None); R = d.iloc[:, 1:].to_numpy(float); R[R == 99] = np.nan; R = R[(~np.isnan(R)).sum(1) >= 50]
elif src == "keller":
    w = pd.read_excel("data/raw/keller/keller_vosshall_2016_MOESM1.xlsx", header=2); w.columns = [str(c).strip() for c in w.columns]
    w = w.rename(columns={"Subject # (this study)": "participant_id", "Odor dilution": "dil", "HOW PLEASANT IS THE SMELL?": "rating", "CID": "image_id"}); w["rating"] = pd.to_numeric(w.rating, errors="coerce")
    w = w[w.dil.isin(("1/10", "1/1,000"))].dropna(subset=["rating"]); R = long_to_mat(w, 50)
V = ~np.isnan(R); N, F = R.shape; print(f"{src}: raters {N} items {F} observed {V.mean():.3f} ratings/rater median {np.median(V.sum(1)):.0f}")
R0 = np.where(V, R, 0.0); im = R0.sum(0) / np.maximum(V.sum(0), 1); rm = R0.sum(1) / np.maximum(V.sum(1), 1)
res = np.where(V, R - im[None] - rm[:, None] + R0.sum() / V.sum(), np.nan)
vt = np.nanvar(R); print(f"  variance share: item {np.var(im[V.sum(0) > 0]) / vt:.3f} rater {np.var(rm) / vt:.3f} residual {np.nanvar(res) / vt:.3f}")
rng = np.random.default_rng(0); fid = rng.permutation(N) % 10; users = rng.permutation(N)[:MAXU]; out = []
for k in users:
    tr = np.where(fid != fid[k])[0]; Vtr = V[tr]; pop = np.where(Vtr.sum(0) > 0, R0[tr].sum(0) / np.maximum(Vtr.sum(0), 1), np.nan)
    Res = np.where(Vtr, R[tr] - pop[None], 0.0); Res = Res / (np.nanstd(Res[Vtr]) + 1e-9)
    avail = np.where(V[k] & ~np.isnan(pop))[0]; g = np.random.default_rng(k)
    for n_est in (20, 50, 100):
        if len(avail) < n_est + 20: continue
        est = g.choice(avail, n_est, replace=False); rest = np.setdiff1d(avail, est)
        for alpha in (10.0, 100.0, 1000.0, 10000.0):
            coef = Ridge(alpha=alpha, fit_intercept=False).fit(Res[:, est].T, (R[k, est] - pop[est]) / (np.nanstd(Res[Vtr]) + 1e-9)).coef_
            pred = pop[rest] + Res[:, rest].T @ coef
            out.append(dict(user=k, n_est=n_est, alpha=alpha, r_pop=np.corrcoef(pop[rest], R[k, rest])[0, 1], r_cf=np.corrcoef(pred, R[k, rest])[0, 1]))
d = pd.DataFrame(out).dropna(); S = d.groupby(["n_est", "alpha"])[["r_pop", "r_cf"]].mean(); S["gain"] = S.r_cf - S.r_pop; best = S.loc[S.groupby(level=0).gain.idxmax()]
print("  CF gain over population mean (best alpha per n_est):"); print(best.round(3).to_string())
