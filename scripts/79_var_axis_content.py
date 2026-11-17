"""T1 (claim revision, 2026-10-29): structure of the per-rater weight variation in the D2 fold-model basis (results/sign_check_<ds>_dims.csv).
Per fold model: centred covariance of the model-free per-rater ridge weights w_k (8 dims) -> first eigenvector v1 (sign: w_pop.v1 > 0);
rater scores s_k = w_k.v1.  Reports: variance share of v1, |cos(v1, w_pop)|, population projection / SD of s, fraction of raters with s < 0,
bimodality coefficient of s, and the content of the axis: Spearman(s_k, content preference) (LAPIS: abs_pref = abstract - figurative z-scored
rating; EVA: residual preference per content category 1-6) and the loading correspondence corr(v1_j, abs_r_j) over the 8 dims (LAPIS).
Outputs results/var_axis_content.csv (per fold model) and figures var_axis_lapis.pdf / var_axis_eva.pdf (distribution of s pooled over fold models).
The two histograms are drawn from the saved bin densities (results/figdata/var_axis_<ds>.csv); FIGDATA=read redraws them without the per-rater files
(the analysis part is then skipped and results/var_axis_content.csv is left as it is).
Run: uv run python -I scripts/79_var_axis_content.py      (FIGDATA=read uv run python -I scripts/79_var_axis_content.py)
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, skew, kurtosis
from face_id.plotting import plt, savefig, esc, TAB10, figdata, FROM_FIGDATA
def bimod(x): n = len(x); return (skew(x) ** 2 + 1) / (kurtosis(x) + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
dist = {}
if not FROM_FIGDATA:
    # EVA content preference per rater (as in 70): residual rating per category
    v = pd.read_csv("data/processed/eva_ratings.csv"); v["image_id"] = v.image_id.astype(str); cat = pd.read_csv("data/raw/eva/image_content_category.csv", dtype=str).rename(columns={"sort": "cat"}); v = v.merge(cat, on="image_id", how="left")
    n = v.groupby("participant_id").size(); v = v[v.participant_id.isin(n[n >= 100].index)]
    v["res"] = v.rating - v.groupby("image_id").rating.transform("mean") - v.groupby("participant_id").rating.transform("mean") + v.rating.mean(); EVAP = v.groupby(["participant_id", "cat"]).res.mean().unstack()
    rows = []; dist = {}
    for ds in ("lapis", "eva"):
        D = pd.read_csv(f"results/sign_check_{ds}_dims.csv"); R = pd.read_csv(f"results/sign_check_{ds}_raters.csv").set_index(["fold", "seed", "rater"]); PD = pd.read_csv(f"results/sign_check_{ds}_perdim.csv")
        wc = [f"w{k}" for k in range(8)]
        for (fold, s), g in D[D.rater.astype(str) != "-1"].groupby(["fold", "seed"]):
            W = g[wc].to_numpy(float); wpop = D[(D.fold == fold) & (D.seed == s) & (D.rater.astype(str) == "-1")][wc].to_numpy(float)[0]
            C = np.cov(W.T); lam, V = np.linalg.eigh(C); lam, V = lam[::-1], V[:, ::-1]; v1 = V[:, 0] * np.sign(wpop @ V[:, 0]); sc = W @ v1
            row = dict(ds=ds, fold=fold, seed=s, n=len(W), var_share1=lam[0] / lam.sum(), part_ratio=lam.sum() ** 2 / (lam ** 2).sum(), cos_v1_pop=abs(v1 @ wpop) / np.linalg.norm(wpop),
                       pop_proj_over_sd=(wpop @ v1) / sc.std(), frac_opposite=(sc < 0).mean(), bimodality=bimod(sc), var_over_pop=np.sqrt(lam.sum()) / np.linalg.norm(wpop), cos_neg=((W @ wpop) < 0).mean())
            key = list(zip(g.fold, g.seed, g.rater))
            if ds == "lapis":
                ap = R.loc[key, "abs_pref"].to_numpy(); row["rho_s_abs"] = spearmanr(sc, ap, nan_policy="omit")[0]; res = W - np.outer(sc, v1); row["rho_resid_abs"] = max(abs(spearmanr(res @ V[:, j], ap, nan_policy="omit")[0]) for j in range(1, 8))
                ar = PD[(PD.fold == fold) & (PD.seed == s)].sort_values("dim").abs_r.to_numpy(); row["r_loading_absr"] = np.corrcoef(v1, ar)[0, 1]
            else:
                pref = EVAP.reindex(g.rater.astype(str).to_numpy()) if EVAP.index.dtype == object else EVAP.reindex(g.rater.to_numpy())
                for c in EVAP.columns: row[f"rho_s_cat{c}"] = spearmanr(sc, pref[c].to_numpy(), nan_policy="omit")[0]
            rows.append(row)
            dist.setdefault(ds, ([], []))[0].extend(sc / sc.std()); dist[ds][1].append((wpop @ v1) / sc.std())   # pooled over fold models (SD units per model)
    T = pd.DataFrame(rows); T.to_csv("results/var_axis_content.csv", index=False)
    num = [c for c in T.columns if c not in ("ds", "fold", "seed")]; print("median over fold models:\n", T.groupby("ds")[num].median().round(3).T.to_string())
    print("\nLAPIS rho(s, abs_pref) per fold model: min %.3f max %.3f; sign-consistent: %s" % (T[T.ds == "lapis"].rho_s_abs.min(), T[T.ds == "lapis"].rho_s_abs.max(), (np.sign(T[T.ds == "lapis"].rho_s_abs).nunique() == 1)))
    ev = T[T.ds == "eva"].filter(like="rho_s_cat"); print("EVA rho(s, category preference) median per category:", ev.median().round(3).to_dict())
BINS = np.linspace(-3.5, 4.5, 33)
for ds, lab in (("lapis", "LAPIS artworks"), ("eva", "EVA photographs")):
    def table():
        sc = np.array(dist[ds][0]); h, _ = np.histogram(np.clip(sc, BINS[0], BINS[-1]), bins=BINS, density=True)
        return pd.DataFrame(dict(bin_left=BINS[:-1], bin_right=BINS[1:], density=h, n_values=len(sc), population_projection_median=float(np.median(dist[ds][1])), frac_opposite=np.mean(sc < 0)))
    H = figdata(f"var_axis_{ds}", table); pp = H.population_projection_median.iloc[0]; fig, ax = plt.subplots()
    ax.hist((H.bin_left + H.bin_right) / 2, bins=BINS, weights=H.density, color=TAB10[0 if ds == "lapis" else 1], alpha=0.8)      # same bars as ax.hist(values, bins=BINS, density=True)
    ax.axvline(0, color="k", lw=0.6, ls=":"); ax.axvline(pp, color="k", lw=0.9); ax.set_xlabel(esc("Weight along the first variation axis (SD units)")); ax.set_ylabel("Density")
    savefig(fig, f"var_axis_{ds}", w=3.4, h=2.6, index_line=f"{lab}: distribution over held-out raters of the model-free per-rater weight projected on the first eigenvector of the between-rater weight covariance (each D2 fold model's basis, SD units per model, pooled over the 20 (LAPIS) / 30 (EVA) fold models; sign such that the population weight projects positively). Solid line = median population weight projection ({pp:.2f} SD), dotted = 0; fraction of raters on the opposite side {H.frac_opposite.iloc[0]:.2f}. From results/var_axis_content.csv (scripts/79).")
print("figures written")
