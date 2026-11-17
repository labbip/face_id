"""Reliability-stratified evaluation: pooled CV results split by participants' split-half reliability
(median split and top/bottom quartiles), for model vs baselines. usage: --exp 1 --tag _cv"""
import argparse, glob, os, sys, json
import numpy as np, pandas as pd
from scipy.stats import ttest_rel
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import RESULTS
ap = argparse.ArgumentParser(); ap.add_argument("--exp", type=int, default=1); ap.add_argument("--tag", default="_cv"); args = ap.parse_args()
rel = pd.read_csv(os.path.join(RESULTS, f"repro_exp{args.exp}_reliability.csv"))[["pid", "split_half_rho", "tr_rho"]]
rows = []
for d in sorted(glob.glob(os.path.join(RESULTS, "runs", f"exp{args.exp}_D*_D8_M*_f*_s*{args.tag}"))):
    if not os.path.exists(os.path.join(d, "eval.csv")): continue
    a = json.load(open(os.path.join(d, "args.json"))); e = pd.read_csv(os.path.join(d, "eval.csv")); e = e[(e.frac_est == 0.7) & (e.infer == "amortized")].copy()
    e["method"] = f"{a['kind']}-M{a['M']}"; rows.append(e[["pid", "method", "rho_hold", "rho_tr_unseen"]])
ev = pd.concat(rows).groupby(["pid", "method"]).mean().reset_index()
bl = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RESULTS, "baselines", f"exp{args.exp}_f*.csv"))])
for b in ["agg", "blend"]:
    ev = pd.concat([ev, bl[["pid", f"rho_hold_{b}", f"rho_tr_unseen_{b}"]].rename(columns={f"rho_hold_{b}": "rho_hold", f"rho_tr_unseen_{b}": "rho_tr_unseen"}).assign(method=b)])
ev = ev.merge(rel, on="pid")
ev["stratum"] = pd.qcut(ev.split_half_rho, [0, 0.25, 0.5, 0.75, 1.0], labels=["Q1 (least reliable)", "Q2", "Q3", "Q4 (most reliable)"])
out = []
for (st, m), g in ev.groupby(["stratum", "method"], observed=True):
    out.append({"stratum": st, "method": m, "n": len(g), "rho_hold": g.rho_hold.mean(), "rho_tr_unseen": g.rho_tr_unseen.mean(), "split_half_mean": g.split_half_rho.mean()})
out = pd.DataFrame(out)
piv = out.pivot(index="method", columns="stratum", values="rho_hold").loc[["agg", "blend", "D0-M4", "D1-M4", "D2-M4", "D3-M4"]]
print("held-out rho by reliability quartile:\n", piv.round(3).to_string())
piv2 = out.pivot(index="method", columns="stratum", values="rho_tr_unseen").loc[["agg", "blend", "D0-M4", "D1-M4", "D2-M4", "D3-M4"]]
print("\nunseen-face test-retest rho by reliability quartile:\n", piv2.round(3).to_string())
# paired model-vs-agg gain within each quartile
best = "D1-M4" if args.exp == 2 else "D2-M4"
for st, g in ev.groupby("stratum", observed=True):
    a = g[g.method == best].set_index("pid"); b = g[g.method == "D0-M4"].set_index("pid"); c = g[g.method == "blend"].set_index("pid"); idx = a.index.intersection(b.index).intersection(c.index)
    print(f"{st}: {best} - D0: hold {np.mean(a.loc[idx].rho_hold - b.loc[idx].rho_hold):+.3f} (p={ttest_rel(a.loc[idx].rho_hold, b.loc[idx].rho_hold).pvalue:.1e}); {best} - blend: hold {np.mean(a.loc[idx].rho_hold - c.loc[idx].rho_hold):+.3f}; unseen {np.nanmean(a.loc[idx].rho_tr_unseen - b.loc[idx].rho_tr_unseen):+.3f}")
out.to_csv(os.path.join(RESULTS, f"reliability_strata_exp{args.exp}{args.tag}.csv"), index=False)
