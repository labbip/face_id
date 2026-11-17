"""Aggregate London Set CV (results/london/eval_*.csv): summary table, paired tests, learning curve,
and perceiver-attribute decoding from z (sex, sexual preference) as the H5 analogue.
Reads the main configuration only (D = 8, M = 4; results/london also holds the D/M sensitivity models, see scripts/45 and 83).
Run: uv run python -I scripts/25_london_analyze.py
"""
import glob, numpy as np, pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/london/eval_*_D8_M4_f*_s*.csv"))], ignore_index=True)
print("runs:", df.groupby(["kind", "seed"]).fold.nunique().to_dict())
met = ["r_pop", "r_amort", "r_direct", "rho_pop", "rho_amort", "rho_direct"]
# per rater x kind x n_est: average over seeds first
g = df.groupby(["kind", "n_est", "rater"])[met].mean().reset_index()
S = g.groupby(["kind", "n_est"])[met].mean().round(3)
S.to_csv("results/london_summary.csv"); print(S.to_string())
# paired comparisons at each n_est (amortised r)
rows = []
def paired(x, y, name, n):
    d = (x - y).dropna(); rows.append(dict(n_est=n, contrast=name, delta=d.mean(), se=d.std() / np.sqrt(len(d)), p_t=ttest_rel(x[d.index], y[d.index]).pvalue, p_w=wilcoxon(d).pvalue, n=len(d)))
for n in sorted(g.n_est.unique()):
    w = g[g.n_est == n].pivot(index="rater", columns="kind", values="r_amort"); wp = g[(g.n_est == n) & (g.kind == "D0")].set_index("rater").r_pop
    wd = g[g.n_est == n].pivot(index="rater", columns="kind", values="r_direct")
    paired(w["D0"], wp.reindex(w.index), "D0 - pop", n)
    for a_, b_ in (("D1", "D0"), ("D2", "D1"), ("D3", "D2"), ("D2", "D0")):
        if a_ in w and b_ in w: paired(w[a_], w[b_], f"{a_} - {b_}", n)
    for k in ("D1", "D2", "D3"):
        if k in wd: paired(w[k], wd[k], f"{k} amort - direct", n)
P = pd.DataFrame(rows); P.to_csv("results/london_paired.csv", index=False)
print(P.round(4).to_string(index=False))
# z -> perceiver attributes (n_est=40 amortised z, averaged over seeds), 10-fold CV balanced accuracy
r = pd.read_csv("data/raw/london/london_faces_ratings.csv"); meta = r.iloc[:, :3]
for kind in ("D1", "D2"):
    sub = df[(df.kind == kind) & (df.n_est == 40)]; zc = [c for c in sub.columns if c.startswith("z") and c[1:].isdigit()]
    if not len(sub): continue
    Z = sub.groupby("rater")[zc].mean(); mt = meta.loc[Z.index]
    for name, y in (("rater_sex (f/m)", mt.rater_sex.where(mt.rater_sex.isin(["female", "male"]))),
                    ("sexpref (men/women)", mt.rater_sexpref.where(mt.rater_sexpref.isin(["men", "women"]))),
                    ("prefers male faces: (f,men)+(m,men) vs (f,women)+(m,women)", mt.rater_sexpref.where(mt.rater_sexpref.isin(["men", "women"])))):
        ok = y.notna(); acc = cross_val_score(LogisticRegression(max_iter=1000, class_weight="balanced"), Z[ok], y[ok], cv=10, scoring="balanced_accuracy")
        print(f"{kind} z -> {name}: balanced acc {acc.mean():.3f} (n={ok.sum()})")
        if name.startswith("sexpref"): break
