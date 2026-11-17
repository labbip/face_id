"""Rebuild the main tables of the two learned-embedding datasets from the main configuration only (D = 8, M = 4, all 5 seeds):
  results/london_{summary,paired}.csv   (London Set, ratings version)
  results/avfs_{summary,paired}.csv     (AVFS novel set, 56 faces)
Why: results/london and results/avfs also hold the D = 5/16 and M = 2/8 sensitivity models (10 folds x 1 seed each), and the earlier tables
were written on 2026-10-20/21 before seeds 2-4 existed. Same columns, contrasts and tests as before (seeds averaged per rater, paired t and
Wilcoxon). The sensitivity models are reported separately in results/generalization_sens.csv (scripts/45).
Run: uv run python -I scripts/83_main_tables_london_avfs.py
"""
import glob, numpy as np, pandas as pd
from scipy.stats import ttest_rel, wilcoxon
MAIN = "eval_*_D8_M4_f*_s*.csv"
def build(src, idcol, met, amort, direct, pop=None):
    fs = sorted(glob.glob(f"results/{src}/{MAIN}")); df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    print(src, "files", len(fs), "runs (folds per kind, seed):", df.groupby(["kind", "seed"]).fold.nunique().to_dict())
    g = df.groupby(["kind", "n_est", idcol])[met].mean().reset_index()          # seeds averaged per rater first
    S = g.groupby(["kind", "n_est"])[met].mean().round(4 if src == "avfs" else 3); S.to_csv(f"results/{src}_summary.csv"); print(S.to_string())
    rows = []
    def paired(x, y, name, n):
        d = (x - y).dropna(); rows.append(dict(n_est=n, contrast=name, delta=d.mean(), se=d.std() / np.sqrt(len(d)), p_t=ttest_rel(x[d.index], y[d.index]).pvalue, p_w=wilcoxon(d).pvalue, n=len(d)))
    for n in sorted(g.n_est.unique()):
        w = g[g.n_est == n].pivot(index=idcol, columns="kind", values=amort); wd = g[g.n_est == n].pivot(index=idcol, columns="kind", values=direct)
        if pop: paired(w["D0"], g[(g.n_est == n) & (g.kind == "D0")].set_index(idcol)[pop].reindex(w.index), "D0 - pop", n)
        for a_, b_ in (("D1", "D0"), ("D2", "D1"), ("D3", "D2"), ("D2", "D0")):
            if a_ in w and b_ in w: paired(w[a_], w[b_], f"{a_} - {b_}", n)
        for k in ("D1", "D2", "D3"):
            if k in wd and wd[k].notna().any(): paired(w[k], wd[k], f"{k} amort - direct", n)
    P = pd.DataFrame(rows); P.to_csv(f"results/{src}_paired.csv", index=False); print(P.round(4).to_string(index=False))
build("london", "rater", ["r_pop", "r_amort", "r_direct", "rho_pop", "rho_amort", "rho_direct"], "r_amort", "r_direct", pop="r_pop")
build("avfs", "ann", ["acc_amort", "acc_direct"], "acc_amort", "acc_direct")
