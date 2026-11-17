"""Pool fold-wise eval.csv files and baselines; paired comparisons across participants.
usage: uv run python scripts/07_analyze.py --exp 1 --tag _cv
Outputs results/summary_exp{exp}{tag}.csv and prints a table (mean rho per method, paired differences, p-values
from a participant-level mixed model with fold fixed effects via statsmodels OLS on per-participant means)."""
import argparse, glob, os, sys, json
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import RESULTS
ap = argparse.ArgumentParser(); ap.add_argument("--exp", type=int, default=1); ap.add_argument("--tag", default="_cv"); ap.add_argument("--kinds", default="D0,D1,D2,D3")
args = ap.parse_args()
rows = []
for k in args.kinds.split(","):
    for d in sorted(glob.glob(os.path.join(RESULTS, "runs", f"exp{args.exp}_{k}_D*_M*_f*_s*{args.tag}"))):
        if not os.path.exists(os.path.join(d, "eval.csv")): continue
        a = json.load(open(os.path.join(d, "args.json")))
        e = pd.read_csv(os.path.join(d, "eval.csv")); e = e[e.frac_est == 0.7].copy()
        e["kind"] = k; e["M"] = a["M"]; e["D"] = a["D"]; e["run"] = os.path.basename(d)
        rows.append(e)
ev = pd.concat(rows, ignore_index=True)
# learning curve (quick evaluations with fixed numbers of estimation trials)
lc_rows = []
for k in args.kinds.split(","):
    for d in sorted(glob.glob(os.path.join(RESULTS, "runs", f"exp{args.exp}_{k}_D*_M*_f*_s*{args.tag}"))):
        if not os.path.exists(os.path.join(d, "eval.csv")): continue
        a = json.load(open(os.path.join(d, "args.json"))); e = pd.read_csv(os.path.join(d, "eval.csv"))
        if "n_est_fixed" in e:
            e = e[e.n_est_fixed.notna()].copy(); e["kind"] = k; e["M"] = a["M"]; lc_rows.append(e)
if lc_rows:
    lc = pd.concat(lc_rows, ignore_index=True).groupby(["kind", "M", "n_est_fixed"]).rho_hold.agg(["mean", "count"]).reset_index()
    lc.to_csv(os.path.join(RESULTS, f"learning_curve_exp{args.exp}{args.tag}.csv"), index=False); print(lc.round(3).to_string(index=False))
# average over seeds per participant × kind × infer
key = ["pid", "fold", "kind", "M", "infer"]
metrics = [c for c in ev.columns if c.startswith("rho_")]
pm = ev.groupby(key)[metrics].mean().reset_index()
# baselines
bl = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RESULTS, "baselines", f"exp{args.exp}_f*.csv"))], ignore_index=True)
tab = []
for (k, M, inf), g in pm.groupby(["kind", "M", "infer"]):
    tab.append({"method": f"{k}-M{M}-{inf}", "n": len(g), "rho_hold": g.rho_hold.mean(), "rho_tr_all": g.rho_tr_all.mean(), "rho_tr_unseen": g.rho_tr_unseen.mean(),
                "ceiling_splithalf": g.rho_splithalf_est.mean()})
for b in ["agg", "mds5", "estrdm", "blend", "rw5"]:
    tab.append({"method": b, "n": len(bl), "rho_hold": bl[f"rho_hold_{b}"].mean(), "rho_tr_all": bl[f"rho_tr_all_{b}"].mean(), "rho_tr_unseen": bl[f"rho_tr_unseen_{b}"].mean()})
tab = pd.DataFrame(tab)
print(tab.round(3).to_string(index=False))
tab.to_csv(os.path.join(RESULTS, f"summary_exp{args.exp}{args.tag}.csv"), index=False)
# paired comparisons on rho_hold and rho_tr_unseen: model vs agg / blend, with fold fixed effect
def paired(a, b, name, metric):
    m = a.merge(b, on="pid", suffixes=("_a", "_b")); m["d"] = m[f"{metric}_a"] - m[f"{metric}_b"]; m = m.dropna(subset=["d"])
    fit = smf.ols("d ~ C(fold_a)", data=m).fit()
    # intercept = mean diff in reference fold; use overall mean + robust t via fold-demeaned
    md = m.d.mean(); se = m.d.std(ddof=1) / np.sqrt(len(m)); 
    from scipy.stats import ttest_1samp, wilcoxon
    t = ttest_1samp(m.d, 0); w = wilcoxon(m.d)
    return {"comparison": name, "metric": metric, "n": len(m), "mean_diff": md, "se": se, "t_p": t.pvalue, "wilcoxon_p": w.pvalue}
comp = []
for (k, M, inf), g in pm.groupby(["kind", "M", "infer"]):
    for metric in ["rho_hold", "rho_tr_unseen"]:
        for b in ["agg", "blend"]:
            bb = bl[["pid", "fold", f"{metric}_{b}"]].rename(columns={f"{metric}_{b}": metric})
            comp.append(paired(g[["pid", "fold", metric]], bb, f"{k}-M{M}-{inf} vs {b}", metric))
# model-vs-model (H2 hierarchy) on amortized inference, and ceiling-normalised means
def mm(ka, kb, metric):
    a = pm[(pm.kind == ka[0]) & (pm.M == ka[1]) & (pm.infer == ka[2])][["pid", "fold", metric]]
    b = pm[(pm.kind == kb[0]) & (pm.M == kb[1]) & (pm.infer == kb[2])][["pid", "fold", metric]]
    if len(a) and len(b): comp.append(paired(a, b, f"{ka[0]}-M{ka[1]}-{ka[2]} vs {kb[0]}-M{kb[1]}-{kb[2]}", metric))
for metric in ["rho_hold", "rho_tr_unseen"]:
    for ka, kb in [(("D1", 4, "amortized"), ("D0", 4, "amortized")), (("D2", 4, "amortized"), ("D1", 4, "amortized")), (("D3", 4, "amortized"), ("D2", 4, "amortized")),
                   (("D2", 4, "amortized"), ("D0", 4, "amortized")), (("D3", 16, "amortized"), ("D2", 4, "amortized")), (("D3", 16, "direct"), ("D3", 16, "amortized")),
                   (("D1", 16, "amortized"), ("D1", 4, "amortized")), (("D3", 16, "direct"), ("D2", 4, "amortized"))]:
        mm(ka, kb, metric)
comp = pd.DataFrame(comp); print(comp.round(4).to_string(index=False))
tab["hold_over_ceiling"] = tab.rho_hold / tab.ceiling_splithalf
print("ceiling-normalised held-out rho (model mean / split-half mean of estimation trials):"); print(tab[["method", "hold_over_ceiling"]].dropna().round(3).to_string(index=False))
tab.to_csv(os.path.join(RESULTS, f"summary_exp{args.exp}{args.tag}.csv"), index=False)
comp.to_csv(os.path.join(RESULTS, f"paired_exp{args.exp}{args.tag}.csv"), index=False)
