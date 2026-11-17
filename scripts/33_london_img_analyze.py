"""Aggregate London Set image-embedding CV (results/london_img/eval_*.csv): seen/unseen-face summary and paired tests.
Run: uv run python -I scripts/33_london_img_analyze.py [london_img|london_hyb]
"""
import glob, sys, numpy as np, pandas as pd
SRC = sys.argv[1] if len(sys.argv) > 1 else "london_img"   # results/<SRC>/eval_*.csv -> results/<SRC>_summary.csv
from scipy.stats import ttest_rel, wilcoxon
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"results/{SRC}/eval_*.csv"))], ignore_index=True)
print("runs:", df.groupby(["kind", "seed"]).fold.nunique().to_dict())
met = [c for c in df.columns if c.startswith("r_")]
g = df.groupby(["kind", "n_est", "rater"])[met].mean().reset_index()
S = g.groupby(["kind", "n_est"])[met].mean().round(3); S.to_csv(f"results/{SRC}_summary.csv"); print(S.to_string())
rows = []
def paired(x, y, name, n, side):
    d = (x - y).dropna(); rows.append(dict(side=side, n_est=n, contrast=name, delta=d.mean(), se=d.std() / np.sqrt(len(d)), p_t=ttest_rel(x[d.index], y[d.index]).pvalue, p_w=wilcoxon(d).pvalue, n=len(d)))
for side in ("unseen", "seen"):
    for n in sorted(g.n_est.unique()):
        w = g[g.n_est == n].pivot(index="rater", columns="kind", values=f"r_amort_{side}"); wd = g[g.n_est == n].pivot(index="rater", columns="kind", values=f"r_direct_{side}")
        base = g[(g.n_est == n) & (g.kind == "D0")].set_index("rater")
        paired(w["D0"], base[f"r_pop_{side}"].reindex(w.index), "D0 - pop(raters)", n, side)
        if side == "unseen" and "r_popimg_unseen" in base: paired(w["D0"], base["r_popimg_unseen"].reindex(w.index), "D0 - pop(image ridge)", n, side); paired(w["D1"], base["r_popimg_unseen"].reindex(w.index), "D1 - pop(image ridge)", n, side)
        for a_, b_ in (("D1", "D0"), ("D2", "D1"), ("D3", "D2"), ("D2", "D0")):
            if a_ in w and b_ in w: paired(w[a_], w[b_], f"{a_} - {b_}", n, side)
        for k in ("D1", "D2", "D3"):
            if k in w and k in wd: paired(w[k], wd[k], f"{k} amort - direct", n, side)
P = pd.DataFrame(rows); P.to_csv(f"results/{SRC}_paired.csv", index=False); print(P.round(4).to_string(index=False))
