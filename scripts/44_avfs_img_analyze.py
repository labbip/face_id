"""Aggregate AVFS main-set image-embedding CV (results/avfs_img/eval_*.csv) and FaceSim3D CVs (results/facesim{2D,3D}).
Run: uv run python -I scripts/44_avfs_img_analyze.py
"""
import glob, numpy as np, pandas as pd
from scipy.stats import ttest_rel, wilcoxon
def paired_table(g, metric, idcol, kinds=("D1", "D2", "D3")):
    rows = []
    for n in sorted(g.n_est.unique()):
        w = g[g.n_est == n].pivot(index=idcol, columns="kind", values=metric)
        for a_, b_ in (("D1", "D0"), ("D2", "D1"), ("D3", "D2"), ("D2", "D0")):
            if a_ in w and b_ in w:
                d = (w[a_] - w[b_]).dropna()
                if len(d) > 2: rows.append(dict(metric=metric, n_est=n, contrast=f"{a_} - {b_}", delta=d.mean(), se=d.std() / np.sqrt(len(d)), p_t=ttest_rel(w[a_][d.index], w[b_][d.index]).pvalue, p_w=wilcoxon(d).pvalue if d.abs().sum() > 0 else 1, n=len(d)))
    return pd.DataFrame(rows)
f = sorted(glob.glob("results/avfs_img/eval_*.csv"))
if f:
    df = pd.concat([pd.read_csv(x) for x in f], ignore_index=True); print("AVFS main (CLIP) runs:", df.groupby(["kind", "seed"]).fold.nunique().to_dict())
    met = [c for c in df.columns if c.startswith("acc") or c.startswith("agree")]
    g = df.groupby(["kind", "n_est", "ann"])[met].mean().reset_index()
    S = g.groupby(["kind", "n_est"])[met].mean().round(4); S.to_csv("results/avfs_img_summary.csv"); print(S.to_string())
    P = pd.concat([paired_table(g, m, "ann") for m in ("acc_main", "acc_novel", "acc_same")]); P.to_csv("results/avfs_img_paired.csv", index=False); print(P.round(4).to_string(index=False))
    # ceiling comparison on the common triplets: model vs leave-one-out modal vote, per annotator
    s = g[g.n_est == g.n_est.max()]
    for k in ("D0", "D1", "D3"):
        w = s[s.kind == k].dropna(subset=["acc_same", "agree_modal_loo"]); d = w.acc_same - w.agree_modal_loo
        print(f"{k}: acc on common triplets {w.acc_same.mean():.4f} vs LOO modal-vote agreement {w.agree_modal_loo.mean():.4f}, delta {d.mean():+.4f} (se {d.std()/np.sqrt(len(d)):.4f}, p={ttest_rel(w.acc_same, w.agree_modal_loo).pvalue:.1e}, n={len(d)})")
for c in ("2D", "3D"):
    f = sorted(glob.glob(f"results/facesim{c}/eval_*.csv"))
    if not f: continue
    df = pd.concat([pd.read_csv(x) for x in f], ignore_index=True); print(f"FaceSim3D {c} runs:", df.groupby(["kind", "seed"]).fold.nunique().to_dict())
    g = df.groupby(["kind", "n_est", "ann"])[["acc_amort", "acc_direct"]].mean().reset_index()
    S = g.groupby(["kind", "n_est"])[["acc_amort", "acc_direct"]].mean().round(4); S.to_csv(f"results/facesim{c}_summary.csv"); print(S.to_string())
    P = paired_table(g, "acc_amort", "ann"); P.to_csv(f"results/facesim{c}_paired.csv", index=False); print(P.round(4).to_string(index=False))
