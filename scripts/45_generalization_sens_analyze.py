"""Seeds 0-4 and D/M sensitivity for the generalization tests (London learned embedding, London image, AVFS 56-face set).
Reads results/london, results/london_img, results/avfs; writes results/generalization_sens.csv.
Run: uv run python -I scripts/45_generalization_sens_analyze.py
"""
import glob, re, numpy as np, pandas as pd
rows = []
def summarize(files, idcol, metric, base_metric, label):
    df = pd.concat([pd.read_csv(x) for x in files], ignore_index=True)
    for (kind, D, M), sub in df.groupby(["kind", "D", "M"]):
        seeds = sorted(sub.seed.unique())
        for n in sorted(sub.n_est.unique()):
            s = sub[sub.n_est == n]; per = s.groupby(idcol)[[metric] + ([base_metric] if base_metric in s else [])].mean()
            rows.append(dict(data=label, kind=kind, D=D, M=M, n_est=n, seeds=len(seeds), folds=s.fold.nunique(), value=per[metric].mean(),
                             delta_vs_pop=(per[metric] - per[base_metric]).mean() if base_metric in per else np.nan, n=len(per)))
summarize(sorted(glob.glob("results/london/eval_*.csv")), "rater", "r_amort", "r_pop", "London learned")
summarize(sorted(glob.glob("results/london_img/eval_*.csv")), "rater", "r_amort_unseen", "r_popimg_unseen", "London image (unseen)")
summarize(sorted(glob.glob("results/avfs/eval_*.csv")), "ann", "acc_amort", "none", "AVFS 56 faces")
S = pd.DataFrame(rows); S.to_csv("results/generalization_sens.csv", index=False)
pd.set_option("display.width", 200)
for label, sub in S.groupby("data"):
    print(f"== {label}"); print(sub[sub.n_est.isin([20, 40, 60])].pivot_table(index=["kind", "D", "M", "seeds"], columns="n_est", values="value").round(4).to_string())
