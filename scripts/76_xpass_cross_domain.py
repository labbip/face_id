"""XPASS-Vis cross-domain analysis: the same 129 raters rated art, fashion and scenery (74/75/55 outputs).
Q1 Is the D1/D2 structure a property of the rater? per-rater gains (D1-D0, D2-D1, n_est=100, mean over seeds) and model-free
   reversal indices (55: cos_pop, gap_free_d1, U_unb) correlated across domains (Spearman), with individual test-retest r as reference.
Q2 Do raters who agree on art also agree on fashion/scenery? (a) model-free: rater x rater Pearson r of rater-centred ratings over
   shared items; (b) model-based: z of all 129 raters under each D1 fold model (100 random training-image ratings), rater x rater
   Euclidean distance; Mantel (Spearman) between domains with rater-permutation p; (c) first CCA correlation between z of two domains.
Q3 Rater attributes from z: ridge/logistic 10-fold CV within each fold model's z (age, TIPI Big Five, interests; gender, nationality),
   permutation p; averaged over fold models.
Outputs results/xpass_cross_{raters,q1,q2,q3}.csv. Run: uv run python -I scripts/76_xpass_cross_domain.py [--n_perm 2000]
"""
import argparse, glob, itertools, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
ap = argparse.ArgumentParser(); ap.add_argument("--n_perm", type=int, default=2000); ap.add_argument("--n_perm_attr", type=int, default=200); ap.add_argument("--nest", type=int, default=100); a = ap.parse_args()
G = ["art", "fashion", "scenery"]; PAIRS = list(itertools.combinations(G, 2)); NEST = a.nest; rng = np.random.default_rng(0)
users = pd.read_csv("data/processed/xpass_users.csv").set_index("participant_id"); retest = pd.read_csv("results/xpass_retest.csv")
# ---------- per-rater table ----------
rows = {}
for g in G:
    ev = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"results/xpass_{g}_img/eval_*.csv"))], ignore_index=True); ev = ev[ev.n_est == NEST]
    m = ev.groupby(["kind", "rater"])[["r_amort_seen", "r_amort_unseen", "r_pop_seen"]].mean().unstack(0)
    T = pd.DataFrame(index=m.index); T["r_D0"] = m["r_amort_seen"]["D0"]; T["r_D1"] = m["r_amort_seen"]["D1"]; T["r_D2"] = m["r_amort_seen"]["D2"]
    T["gain_D1_D0"] = T.r_D1 - T.r_D0; T["gain_D2_D1"] = T.r_D2 - T.r_D1; T["gain_D2_D1_unseen"] = m["r_amort_unseen"]["D2"] - m["r_amort_unseen"]["D1"]; T["gain_D1_D0_unseen"] = m["r_amort_unseen"]["D1"] - m["r_amort_unseen"]["D0"]
    S = pd.read_csv(f"results/sign_check_xpass_{g}_raters.csv").groupby("rater")[["cos_pop", "gap_free_d1", "U_unb", "halves_r"]].mean(); T = T.join(S)
    T["retest"] = retest[retest.genre == g].set_index("participant_id")["r_Aesthetic"]; T.columns = [f"{g}_{c}" for c in T.columns]; rows[g] = T
RT = pd.concat(rows.values(), axis=1); RT.index.name = "participant_id"; RT.to_csv("results/xpass_cross_raters.csv")
print("per-rater means:\n", RT.mean().round(3).to_string())
q1 = []
for c in ["gain_D1_D0", "gain_D2_D1", "gain_D2_D1_unseen", "r_D1", "cos_pop", "gap_free_d1", "U_unb", "halves_r", "retest"]:
    for g1, g2 in PAIRS:
        x, y = RT[f"{g1}_{c}"], RT[f"{g2}_{c}"]; ok = x.notna() & y.notna(); r, p = spearmanr(x[ok], y[ok]); q1.append(dict(measure=c, a=g1, b=g2, n=int(ok.sum()), spearman=r, p=p))
Q1 = pd.DataFrame(q1); Q1.to_csv("results/xpass_cross_q1.csv", index=False); print("\nQ1 cross-domain Spearman of per-rater measures:\n", Q1.round(3).to_string(index=False))
# ---------- data + models ----------
class Enc(nn.Module):
    def __init__(s, D, M, h):
        super().__init__(); s.net = nn.Sequential(nn.Linear(2 * D + 1, h), nn.GELU(), nn.Linear(h, h), nn.GELU()); s.att = nn.Linear(h, 1); s.head = nn.Linear(h, M)
    def forward(s, f, i, rz, kidx, n):
        fe = f[i]; x = torch.cat([fe, rz[:, None], rz[:, None] * fe], -1); h = s.net(x)
        al = s.att(h).squeeze(-1); al = al - al.max(); w = torch.exp(al); den = torch.zeros(n).index_add_(0, kidx, w) + 1e-12; w = w / den[kidx]
        return s.head(torch.zeros(n, h.shape[1]).index_add_(0, kidx, h * w[:, None]))
class M_(nn.Module):
    def __init__(s): super().__init__(); s.P = nn.Linear(64, 8); s.enc = Enc(8, 4, 64)
def load_domain(g):
    d = pd.read_csv(f"data/processed/xpass_{g}.csv").groupby(["participant_id", "image_id"], as_index=False).rating.mean()
    ids = np.load(f"results/xpass_{g}_image_ids.npy", allow_pickle=True); imap = {str(m): i for i, m in enumerate(ids)}; d["i"] = d.image_id.astype(str).map(imap); d = d.dropna(subset=["i"]); d["i"] = d.i.astype(int)
    raters = sorted(d.participant_id.unique()); d["k"] = d.participant_id.map({r: i for i, r in enumerate(raters)}); Fn = len(ids)
    ifold = np.random.default_rng(1).permutation(Fn) % 5; d["if"] = ifold[d.i]; E = np.load(f"results/xpass_{g}_embed_clip.npy"); return d, raters, ifold, E
def encode_all(g, d, raters, ifold, E, fold, seed, n_est=NEST, sd_seed=0):
    sd = torch.load(f"results/xpass_{g}_img/model_D1_D8_M4_f{fold}_s{seed}.pt", map_location="cpu"); m = M_(); m.load_state_dict({k: v for k, v in sd.items() if k.startswith(("P.", "enc."))}); m.eval()
    itr = np.where(ifold != fold % 5)[0]; pca = PCA(64, random_state=0).fit(E[itr]); X = pca.transform(E); X = (X - X[itr].mean(0)) / X[itr].std(0)
    f = m.P(torch.tensor(X, dtype=torch.float32)); q = d[d["if"] != fold % 5].sample(frac=1, random_state=sd_seed); q = q[q.groupby("k").cumcount() < n_est]
    mu = q.groupby("k").rating.transform("mean"); sdv = q.groupby("k").rating.transform("std") + 1e-6; rz = torch.tensor(((q.rating - mu) / sdv).to_numpy(), dtype=torch.float32)
    with torch.no_grad(): z = m.enc(f, torch.tensor(q.i.to_numpy()), rz, torch.tensor(q.k.to_numpy()), len(raters)).numpy()
    return pd.DataFrame(z, index=raters, columns=[f"z{j}" for j in range(4)])
def rater_sim(d):     # model-free rater x rater Pearson r over shared items (rater-centred ratings), NaN if < 10 shared
    piv = d.pivot(index="k", columns="i", values="rating"); piv = piv.sub(piv.mean(1), axis=0); A = piv.to_numpy(); n = A.shape[0]; S = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            ok = ~np.isnan(A[i]) & ~np.isnan(A[j])
            if ok.sum() >= 10 and A[i, ok].std() > 0 and A[j, ok].std() > 0: S[i, j] = S[j, i] = np.corrcoef(A[i, ok], A[j, ok])[0, 1]
    return S
def mantel(A, B, n_perm):
    iu = np.triu_indices(len(A), 1); x, y = A[iu], B[iu]; ok = ~np.isnan(x) & ~np.isnan(y); r = spearmanr(x[ok], y[ok])[0]; cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(len(A)); Bp = B[np.ix_(p, p)]; yp = Bp[iu]; okp = ~np.isnan(x) & ~np.isnan(yp); cnt += spearmanr(x[okp], yp[okp])[0] >= r
    return r, (cnt + 1) / (n_perm + 1), int(ok.sum())
def cca1(Zx, Zy, n_perm):
    def c1(X, Y): cc = CCA(1).fit(X, Y); u, v = cc.transform(X, Y); return pearsonr(u[:, 0], v[:, 0])[0]
    r = c1(Zx, Zy); null = np.array([c1(Zx, Zy[rng.permutation(len(Zy))]) for _ in range(n_perm)]); return r, (np.sum(null >= r) + 1) / (n_perm + 1)
D = {g: load_domain(g) for g in G}; assert all(D[g][1] == D[G[0]][1] for g in G), "rater sets differ"
SIM = {g: rater_sim(D[g][0]) for g in G}; q2 = []
for g1, g2 in PAIRS:
    r, p, n = mantel(SIM[g1], SIM[g2], a.n_perm); q2.append(dict(test="model-free rater similarity (Mantel)", a=g1, b=g2, fold=-1, seed=-1, r=r, p=p, n_pairs=n)); print(f"Q2a {g1}-{g2}: Mantel rho {r:+.3f} p={p:.4f} (pairs {n})", flush=True)
for g in G:     # within-domain reliability of the model-free similarity: split items in two halves
    d = D[g][0]; h = np.random.default_rng(5).permutation(d.i.max() + 1) % 2; S1, S2 = rater_sim(d[h[d.i] == 0]), rater_sim(d[h[d.i] == 1]); r, p, n = mantel(S1, S2, 200); q2.append(dict(test="model-free similarity split-half (items)", a=g, b=g, fold=-1, seed=-1, r=r, p=p, n_pairs=n)); print(f"Q2a {g} split-half Mantel rho {r:+.3f} (pairs {n})", flush=True)
Z = {}
for g in G:
    for fold in range(10):
        for seed in (0, 1): Z[(g, fold, seed)] = encode_all(g, *D[g], fold, seed)
for fold in range(10):
    for seed in (0, 1):
        for g1, g2 in PAIRS:
            Z1, Z2 = Z[(g1, fold, seed)], Z[(g2, fold, seed)]; A = np.linalg.norm(Z1.to_numpy()[:, None] - Z1.to_numpy()[None], axis=-1); B = np.linalg.norm(Z2.to_numpy()[:, None] - Z2.to_numpy()[None], axis=-1)
            r, p, n = mantel(A, B, 200); q2.append(dict(test="z distance (Mantel)", a=g1, b=g2, fold=fold, seed=seed, r=r, p=p, n_pairs=n))
            r, p = cca1(Z1.to_numpy(), Z2.to_numpy(), 200); q2.append(dict(test="z CCA first correlation", a=g1, b=g2, fold=fold, seed=seed, r=r, p=p, n_pairs=len(Z1)))
        for g in G:   # z reliability within domain: same fold model, two disjoint 100-rating subsets -> Mantel of z distances
            Za, Zb = encode_all(g, *D[g], fold, seed, sd_seed=11), encode_all(g, *D[g], fold, seed, sd_seed=12)
            A = np.linalg.norm(Za.to_numpy()[:, None] - Za.to_numpy()[None], axis=-1); B = np.linalg.norm(Zb.to_numpy()[:, None] - Zb.to_numpy()[None], axis=-1); r, p, n = mantel(A, B, 50); q2.append(dict(test="z distance split-half (ratings)", a=g, b=g, fold=fold, seed=seed, r=r, p=p, n_pairs=n))
Q2 = pd.DataFrame(q2); Q2.to_csv("results/xpass_cross_q2.csv", index=False)
print("\nQ2 summary (mean over fold models):\n", Q2.groupby(["test", "a", "b"]).agg(r=("r", "mean"), r_min=("r", "min"), r_max=("r", "max"), p_median=("p", "median"), frac_p05=("p", lambda x: (x < 0.05).mean())).round(3).to_string())
# ---------- Q3 attributes from z ----------
cont = ["age", "tipi_E", "tipi_A", "tipi_C", "tipi_N", "tipi_O", "art_interest", "fashion_interest", "photoVideo_interest"]; cat = {"gender": ("male", "female"), "nationality": ("JPN", "CHN")}
q3 = []
for g in G:
    for fold in range(10):
        for seed in (0, 1):
            Zm = Z[(g, fold, seed)].loc[users.index]; X = Zm.to_numpy()
            for c in cont:
                y = users[c].to_numpy(float); ok = ~np.isnan(y); mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 11)))
                pr = cross_val_predict(mdl, X[ok], y[ok], cv=KFold(10, shuffle=True, random_state=0)); r = pearsonr(pr, y[ok])[0]
                null = [pearsonr(cross_val_predict(mdl, X[ok], rng.permutation(y[ok]), cv=KFold(10, shuffle=True, random_state=0)), y[ok])[0] for _ in range(a.n_perm_attr if (fold, seed) == (0, 0) else 0)]
                q3.append(dict(genre=g, fold=fold, seed=seed, attr=c, score="cv_r", value=r, p_perm=((np.sum(np.array(null) >= r) + 1) / (len(null) + 1)) if null else np.nan))
            for c, (l0, l1) in cat.items():
                y = users[c].to_numpy(); ok = np.isin(y, [l0, l1]); yb = (y[ok] == l1).astype(int); mdl = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))
                pr = cross_val_predict(mdl, X[ok], yb, cv=StratifiedKFold(10, shuffle=True, random_state=0)); r = balanced_accuracy_score(yb, pr)
                null = [balanced_accuracy_score(yb, cross_val_predict(mdl, X[ok], rng.permutation(yb), cv=StratifiedKFold(10, shuffle=True, random_state=0))) for _ in range(a.n_perm_attr if (fold, seed) == (0, 0) else 0)]
                q3.append(dict(genre=g, fold=fold, seed=seed, attr=c, score="balanced_acc", value=r, p_perm=((np.sum(np.array(null) >= r) + 1) / (len(null) + 1)) if null else np.nan))
Q3 = pd.DataFrame(q3); Q3.to_csv("results/xpass_cross_q3.csv", index=False)
print("\nQ3 attribute decoding from z (mean over 20 fold models; permutation p from fold 0 seed 0):\n", Q3.groupby(["attr", "score", "genre"]).agg(value=("value", "mean"), sd=("value", "std"), p_f0s0=("p_perm", "first")).round(3).unstack("genre").to_string())
