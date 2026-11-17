"""T3 (claim revision, 2026-10-29): LAPIS-like synthetic raters with KNOWN weight structure on the real LAPIS CLIP features, to test whether
the apparatus (53 D0/D1/D2 + 55 sign check) reproduces the LAPIS signature (D2-D1 +0.021, D1-constraint loss +0.046, gap>0.02 52%, cos<0 9.6%).
Stimulus basis X: PCA-64 of CLIP (all 11,723 images), standardised.  Population weight w_pop: ridge of pooled per-rater z-scored LAPIS ratings on X.
Content axis u_abs: unit ridge direction of the abstract indicator; u_perp = its component orthogonal to w_pop.
Conditions (W0 = |w_pop|, a, b ~ N(0,1) per rater, u1..u3 random unit vectors orthogonal to w_pop and u_perp):
  cont : w_k = w_pop + W0 (0.95 a_k v1 + 0.45 b1 u1 + 0.35 b2 u2 + 0.30 b3 u3), v1 = 0.59 w_pop/W0 - 0.81 u_perp (~ -u_abs)
         -> first variation axis = abstract axis with |cos(v1, w_pop)| 0.59, population projection / SD 0.62, 27% opposite, unimodal (LAPIS values, 79)
  disc : w_k = w_pop + W0 0.25 (b1 u1 + b2 u2 + b3 u3); for a random 10% the component along u_abs is reflected (w_k -= 2 (w_k.u_abs) u_abs)
         -> discrete minority reversal of the abstract axis (the structure the successor project found undetectable by D2-D1)
Ratings: y = x.w_k + sigma n, 500 raters x 480 random images; sigma calibrated so that the per-rater correlation with the leave-one-out image mean
matches LAPIS (computed here with the same code); the split-half reliability of per-rater ridge weights (16 PCs) is reported for both.
Outputs data/processed/sim_lapis_<cond>.csv, results/sim_lapis_like/<cond>_truth.csv, results/sim_lapis_like/calibration.csv
Run: uv run python -I scripts/81_sim_lapis_like.py [--sigma_grid 0.6,0.8,1.0,1.2]
"""
import argparse, os, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
ap = argparse.ArgumentParser(); ap.add_argument("--sigma_grid", default="0.5,0.7,0.9,1.1,1.4"); ap.add_argument("--n_raters", type=int, default=500); ap.add_argument("--n_rat", type=int, default=480); a = ap.parse_args()
os.makedirs("results/sim_lapis_like", exist_ok=True); rng = np.random.default_rng(0)
ids = np.load("results/lapis_image_ids.npy", allow_pickle=True); E = np.load("results/lapis_embed_clip.npy"); X = PCA(64, random_state=0).fit_transform(E); X = (X - X.mean(0)) / X.std(0); Fn = len(ids); pos = {int(i): j for j, i in enumerate(ids)}
d = pd.read_csv("data/raw/lapis/piaa.csv", usecols=["participant_id", "image_id", "rating", "2_styles"], low_memory=False); sty = d.drop_duplicates("image_id").set_index("image_id")["2_styles"]
d = d.groupby(["participant_id", "image_id"], as_index=False).rating.mean(); n = d.groupby("participant_id").size(); d = d[d.participant_id.isin(n[n >= 100].index)]; d["i"] = d.image_id.map(pos)
d["rz"] = (d.rating - d.groupby("participant_id").rating.transform("mean")) / d.groupby("participant_id").rating.transform("std")
w_pop = Ridge(alpha=10.0).fit(X[d.i], d.rz).coef_; W0 = np.linalg.norm(w_pop); wh = w_pop / W0
abstract = (sty.reindex(ids).to_numpy() == "ABSTRACT").astype(float); u_abs = Ridge(alpha=10.0).fit(X, abstract).coef_; u_abs /= np.linalg.norm(u_abs)
u_perp = u_abs - (u_abs @ wh) * wh; u_perp /= np.linalg.norm(u_perp); print(f"|w_pop| {W0:.3f}; cos(u_abs, w_pop) {u_abs @ wh:+.3f} (population prefers figurative if negative)")
def orth_basis(k, against):
    B = []; A = np.array(against)
    while len(B) < k:
        v = rng.standard_normal(64); v -= A.T @ (A @ v); v -= sum((v @ b) * b for b in B); v /= np.linalg.norm(v); B.append(v)
    return np.array(B)
U = orth_basis(3, [wh, u_perp]); v1 = 0.59 * wh - 0.81 * u_perp   # sign: +v1 = population side AND more figurative (LAPIS: rho(s, abs_pref) = -0.93, 79); cos(u_abs, w_pop) = -0.76 so this makes v1 ~ -u_abs
print(f"cos(v1, u_abs) {v1 @ u_abs:+.3f}")
def consensus_and_reliability(df):   # per-rater r with leave-one-out image mean; split-half r of per-rater ridge weights (16 PCs)
    s = df.groupby("i").rating.agg(["sum", "count"]); q = df.join(s, on="i"); q["loo"] = (q["sum"] - q.rating) / (q["count"] - 1).replace(0, np.nan)
    cons = q.groupby("participant_id").apply(lambda g: np.corrcoef(g.rating, g.loo)[0, 1]).mean(); rel = []
    for k, g in df.groupby("participant_id"):
        g = g.sample(frac=1, random_state=1); h = len(g) // 2; y = (g.rating - g.rating.mean()).to_numpy()
        wa = Ridge(alpha=30.0).fit(X[g.i.to_numpy()[:h], :16], y[:h]).coef_; wb = Ridge(alpha=30.0).fit(X[g.i.to_numpy()[h:], :16], y[h:]).coef_; rel.append(np.corrcoef(wa, wb)[0, 1])
    return cons, float(np.mean(rel))
L = d.rename(columns={"rz": "_"}); c_l, r_l = consensus_and_reliability(L); print(f"LAPIS: consensus r {c_l:.3f}, ridge-weight split-half r {r_l:.3f}")
def make(cond, sigma, seed=0):
    g = np.random.default_rng(100 + seed); N = a.n_raters; A = g.standard_normal(N); B = g.standard_normal((N, 3)); refl = np.zeros(N, bool)
    if cond == "cont": W = w_pop[None] + W0 * (0.95 * A[:, None] * v1[None] + (B * np.array([0.45, 0.35, 0.30])[None]) @ U)
    else:
        W = w_pop[None] + W0 * 0.25 * (B @ U); refl[g.choice(N, N // 10, replace=False)] = True; W[refl] -= 2 * (W[refl] @ u_abs)[:, None] * u_abs[None]
    rows = []
    for k in range(N):
        ii = g.choice(Fn, a.n_rat, replace=False); y = X[ii] @ W[k] + sigma * g.standard_normal(a.n_rat); rows.append(pd.DataFrame(dict(participant_id=k, image_id=ids[ii], i=ii, rating=y)))
    df = pd.concat(rows, ignore_index=True); truth = pd.DataFrame(dict(rater=np.arange(N), a=A, reflected=refl, abs_readout=W @ u_abs, pop_readout=W @ wh, cos_pop=(W @ wh) / np.linalg.norm(W, axis=1)))
    return df, truth
cal = []
for cond in ("cont", "disc"):
    for sigma in [float(x) for x in a.sigma_grid.split(",")]:
        df, _ = make(cond, sigma); c, r = consensus_and_reliability(df); cal.append(dict(cond=cond, sigma=sigma, consensus=c, reliability=r)); print(f"{cond} sigma {sigma}: consensus {c:.3f} reliability {r:.3f}", flush=True)
C = pd.DataFrame(cal); C["target_consensus"] = c_l; C["target_reliability"] = r_l; C.to_csv("results/sim_lapis_like/calibration.csv", index=False)
for cond in ("cont", "disc"):
    q = C[C.cond == cond]; sigma = float(q.sigma.iloc[(q.consensus - c_l).abs().argmin()]); df, truth = make(cond, sigma)
    df["rating"] = 50 + 20 * (df.rating - df.rating.mean()) / df.rating.std(); df[["participant_id", "image_id", "rating"]].to_csv(f"data/processed/sim_lapis_{cond}.csv", index=False); truth.to_csv(f"results/sim_lapis_like/{cond}_truth.csv", index=False)
    print(f"{cond}: sigma {sigma} -> data/processed/sim_lapis_{cond}.csv ({len(df)} ratings); true abstract-readout opposite to population: {(np.sign(truth.abs_readout) != np.sign(w_pop @ u_abs)).mean():.3f}; true cos<0: {(truth.cos_pop < 0).mean():.3f}")
