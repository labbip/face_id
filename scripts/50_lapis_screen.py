"""LAPIS screening (art aesthetics, 555 raters x ~500 of 11,723 images each): is there systematic idiosyncratic variance?
(1) per-rater style preference (abstract minus figurative residual) and its split-half reliability across the rater's own images;
(2) CV: predict a rater's held-out ratings by image mean (other raters) vs image mean + low-rank residual model (ALS, rank k)
    fitted on the rater's other ratings and all other raters' ratings.
Run: uv run python -I scripts/50_lapis_screen.py
"""
import numpy as np, pandas as pd
from scipy.stats import pearsonr
rng = np.random.default_rng(0)
d = pd.read_csv("data/raw/lapis/piaa.csv", low_memory=False)[["image_id", "rating", "participant_id", "2_styles", "style", "Art Interest VAIAK"]]
d = d.groupby(["participant_id", "image_id"], as_index=False).agg(rating=("rating", "mean"), s2=("2_styles", "first"), art=("Art Interest VAIAK", "first"))
n = d.groupby("participant_id").size(); d = d[d.participant_id.isin(n[n >= 100].index)].reset_index(drop=True)
P = {p: i for i, p in enumerate(sorted(d.participant_id.unique()))}; I = {m: i for i, m in enumerate(sorted(d.image_id.unique()))}
d["p"], d["i"] = d.participant_id.map(P), d.image_id.map(I); NP, NI = len(P), len(I)
print(f"raters {NP} (>=100 ratings), images {NI}, ratings {len(d)}")
d["test"] = rng.random(len(d)) < 0.3
tr, te = d[~d.test], d[d.test]
# rater and image means from training cells (image mean excludes the rater by construction of the split only partially; use leave-one-out below)
mu = tr.rating.mean(); rm = tr.groupby("p").rating.mean() - mu
img_sum = tr.groupby("i").rating.sum(); img_cnt = tr.groupby("i").size()
# baseline for a test cell (p,i): image mean over training raters (rater p's own training rating of i does not exist since cells are unique) + rater mean
te = te.assign(pred_img=lambda x: (img_sum.reindex(x.i).to_numpy() / img_cnt.reindex(x.i).to_numpy()) + rm.reindex(x.p).to_numpy() - mu + mu)
te = te.dropna(subset=["pred_img"])
# residual low-rank model: r_pi - mu - a_p - b_i ~= u_p . v_i, ALS with ridge
tr = tr.assign(res=lambda x: x.rating - mu - rm.reindex(x.p).to_numpy() - (img_sum.reindex(x.i).to_numpy() / img_cnt.reindex(x.i).to_numpy() - mu))
def als(k, lam=5.0, iters=15):
    U = rng.normal(0, 0.1, (NP, k)); V = rng.normal(0, 0.1, (NI, k)); pp, ii, rr = tr.p.to_numpy(), tr.i.to_numpy(), tr.res.to_numpy()
    for _ in range(iters):
        for side, idx, other, Xo in (("U", pp, ii, V), ("V", ii, pp, U)):
            A = np.zeros((NP if side == "U" else NI, k, k)); b = np.zeros((NP if side == "U" else NI, k))
            X = Xo[other]; np.add.at(A, idx, X[:, :, None] * X[:, None, :]); np.add.at(b, idx, X * rr[:, None])
            sol = np.linalg.solve(A + lam * np.eye(k)[None], b[:, :, None])[:, :, 0]
            if side == "U": U = sol
            else: V = sol
    return U, V
def per_rater_r(pred):
    out = []
    for p, g in te.assign(pred=pred).groupby("p"):
        if len(g) >= 20 and g.rating.std() > 0 and g.pred.std() > 0: out.append((p, pearsonr(g.pred, g.rating)[0]))
    return pd.Series(dict(out))
base = per_rater_r(te.pred_img.to_numpy()); print(f"image-mean baseline: per-rater r on held-out ratings {base.mean():.3f} (n={len(base)})")
rows = []
for k in (1, 2, 4, 8):
    U, V = als(k); pred = te.pred_img.to_numpy() + (U[te.p.to_numpy()] * V[te.i.to_numpy()]).sum(1)
    r = per_rater_r(pred); dlt = (r - base).dropna(); rows.append(dict(k=k, r=r.mean(), delta=dlt.mean(), se=dlt.std() / np.sqrt(len(dlt))))
    print(f"rank-{k} residual model: r {r.mean():.3f}, delta vs baseline {dlt.mean():+.4f} (se {dlt.std()/np.sqrt(len(dlt)):.4f})")
pd.DataFrame(rows).to_csv("results/lapis_screen.csv", index=False)
# style preference reliability: split each rater's images in two halves; abstract-minus-figurative residual mean in each half
d2 = d.assign(res=lambda x: x.rating - x.groupby("i").rating.transform("mean") - x.groupby("p").rating.transform("mean") + x.rating.mean(), half=rng.integers(0, 2, len(d)))
sp = d2.groupby(["p", "half", "s2"]).res.mean().unstack("s2"); pref = (sp["abstract"] - sp["figurative"]).unstack("half").dropna()
print(f"abstract-vs-figurative preference: split-half r across raters {pearsonr(pref[0], pref[1])[0]:.3f} (n={len(pref)}), SD of preference {pref.mean(1).std():.1f} rating points")
# residual variance share and PCs via dense sub-block? skip; report residual share
tot = ((d.rating - d.rating.mean()) ** 2).sum(); res_share = (d2.res ** 2).sum() / tot; print(f"residual share of variance (after rater and image means) {res_share:.3f}")
