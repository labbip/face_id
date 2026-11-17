"""London Set screening: how much systematic idiosyncratic variance is there in
attractiveness ratings (2,513 raters x 102 faces), and is it predictable from a few faces?

Outputs results/london_screen.csv (+ printed summary).
Run: uv run python -I scripts/22_london_screen.py
"""
import numpy as np, pandas as pd, sys
from numpy.linalg import svd
rng = np.random.default_rng(0)
r = pd.read_csv("data/raw/london/london_faces_ratings.csv")
meta = r.iloc[:, :3].copy(); R = r.iloc[:, 3:].to_numpy(float)       # (N, F)
info = pd.read_csv("data/raw/london/london_faces_info.csv", dtype={"face_id": str})
N, F = R.shape
print(f"raters {N}, faces {F}")

# 1. variance decomposition (two-way, no interaction/noise separation possible: 1 rating per cell)
mu = R.mean(); a = R.mean(1, keepdims=True) - mu; b = R.mean(0, keepdims=True) - mu
res = R - mu - a - b
tot = ((R - mu) ** 2).sum()
print(f"variance share: rater mean {((a**2).sum()*F)/tot:.3f}, face mean {((b**2).sum()*N)/tot:.3f}, residual (idiosyncratic+noise) {(res**2).sum()/tot:.3f}")

# 2. agreement: mean pairwise Pearson r across faces (random 2000 pairs), and rater-vs-leave-one-out-mean r
sd = R.std(1, keepdims=True); print(f"constant raters: {(sd[:,0]==0).sum()}")
Z = (R - R.mean(1, keepdims=True)) / np.where(sd > 0, sd, np.nan)
ii = rng.integers(0, N, 4000); jj = rng.integers(0, N, 4000); m = ii != jj
pair_r = (Z[ii[m]] * Z[jj[m]]).mean(1); pair_r = pair_r[np.isfinite(pair_r)]
loo = (R.sum(0, keepdims=True) - R) / (N - 1)
r_loo = np.array([np.corrcoef(R[k], loo[k])[0, 1] for k in range(N) if sd[k,0] > 0])
print(f"mean pairwise r {pair_r.mean():.3f} (median {np.median(pair_r):.3f}); mean r(rater, others' mean) {r_loo.mean():.3f}")

# 3. is residual structure systematic? rank of residual via split of raters: correlate face-loadings
#    of residual SVD computed on two random halves of raters.
perm = rng.permutation(N); h1, h2 = perm[: N // 2], perm[N // 2 :]
def loadings(idx, k=8):
    X = res[idx]; U, S, Vt = svd(X, full_matrices=False); return Vt[:k], S[:k] ** 2 / (X ** 2).sum()
V1, s1 = loadings(h1); V2, s2 = loadings(h2)
C = np.abs(V1 @ V2.T)
print("residual PCs: var share half1", np.round(s1[:5], 3), " |cos| with half2 PCs (matched):", np.round(C.max(1)[:5], 2))

# 4. CV predictability: held-out raters, 20 estimation faces -> predict 82 others.
#    baselines: population mean (train raters) with per-rater affine fit on est faces  [= D0]
#    rank-M: face loadings from train-rater residual SVD; ridge-fit rater weights on est faces [= free linear, "direct" D2]
#    kNN: 50 nearest train raters by est-face correlation, average their ratings (collaborative filtering)
def cv(n_est=20, Ms=(1, 2, 4, 8, 16), folds=10, ridge=1.0):
    out = []
    fid = rng.permutation(N) % folds
    for f in range(folds):
        tr, te = np.where(fid != f)[0], np.where(fid == f)[0]
        pop = R[tr].mean(0)                                    # (F,)
        resid_tr = R[tr] - R[tr].mean(1, keepdims=True) - pop + R[tr].mean()
        U, S, Vt = svd(resid_tr, full_matrices=False)
        for k in te:
            est = rng.choice(F, n_est, replace=False); mask = np.zeros(F, bool); mask[est] = True
            y = R[k]
            # D0: affine of population
            A = np.c_[np.ones(n_est), pop[est]]; coef = np.linalg.lstsq(A, y[est], rcond=None)[0]
            pred0 = coef[0] + coef[1] * pop
            row = dict(fold=f, rater=k, rho_pop=np.corrcoef(pred0[~mask], y[~mask])[0, 1],
                       mae_pop=np.abs(pred0[~mask] - y[~mask]).mean())
            for M in Ms:
                V = Vt[:M].T                                   # (F, M)
                A = np.c_[np.ones(n_est), pop[est], V[est]]
                lam = np.diag([0, 0] + [ridge] * M)
                coef = np.linalg.solve(A.T @ A + lam, A.T @ y[est])
                pred = coef[0] + coef[1] * pop + V @ coef[2:]
                row[f"rho_M{M}"] = np.corrcoef(pred[~mask], y[~mask])[0, 1]
                row[f"mae_M{M}"] = np.abs(pred[~mask] - y[~mask]).mean()
            # kNN CF on residuals
            ztr = resid_tr[:, est]; zk = y[est] - y[est].mean() - pop[est] + pop.mean()
            sim = (ztr @ zk) / (np.linalg.norm(ztr, axis=1) * np.linalg.norm(zk) + 1e-9)
            nn = np.argsort(-sim)[:50]
            pred = pred0 + resid_tr[nn].mean(0)
            row["rho_knn"] = np.corrcoef(pred[~mask], y[~mask])[0, 1]
            out.append(row)
    return pd.DataFrame(out)

rows = []
for n_est in (5, 10, 20, 40):
    df = cv(n_est=n_est); s = df.drop(columns=["fold", "rater"]).mean(); s["n_est"] = n_est; rows.append(s)
    d = df["rho_M4"] - df["rho_pop"]
    print(f"n_est={n_est}: rho pop {s.rho_pop:.3f} | M1 {s.rho_M1:.3f} M2 {s.rho_M2:.3f} M4 {s.rho_M4:.3f} M8 {s.rho_M8:.3f} M16 {s.rho_M16:.3f} | kNN {s.rho_knn:.3f} | M4-pop {d.mean():+.3f} (se {d.std()/np.sqrt(len(d)):.3f})")
S = pd.DataFrame(rows); S.to_csv("results/london_screen.csv", index=False)

# 5. does a known perceiver attribute (sex x sexual preference) explain residuals? face-gender preference index
fg = (info.face_gender.to_numpy() == "male")
pref_male = res[:, fg].mean(1) - res[:, ~fg].mean(1)
meta["pref_male_faces"] = pref_male
print(meta.groupby(["rater_sex", "rater_sexpref"])["pref_male_faces"].agg(["mean", "count"]).round(2))
